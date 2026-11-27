"""页级运行时池：CPU 并发辅助函数与显存感知的 GPU worker 准入。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
from typing import Any, Callable

from .pipeline import PageParseResult, ParserConfig


@dataclass(frozen=True)
class GpuMemorySnapshot:
    device_index: int
    free_mb: int
    total_mb: int


def gpu_memory_snapshot(device_index: int | None = None) -> GpuMemorySnapshot | None:
    """读取物理 GPU 的空闲显存；查询不可用时返回 None，而不是猜一个数。"""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    if device_index is None:
        override = os.environ.get("PADDLEOCR_GPU_MEMORY_DEVICE")
        device = os.environ.get("PADDLEOCR_VLM_DEVICE", "gpu:0")
        try:
            device_index = int(override) if override is not None else int(device.split(":", 1)[1])
        except (ValueError, IndexError):
            return None
    command = [
        executable,
        f"--id={device_index}",
        "--query-gpu=index,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        if completed.returncode != 0:
            return None
        row = completed.stdout.strip().splitlines()[0]
        index, free, total = [int(item.strip()) for item in row.split(",")]
        return GpuMemorySnapshot(device_index=index, free_mb=free, total_mb=total)
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def resolve_cpu_workers(configured: int, job_count: int) -> int:
    """纯 CPU 阶段直接并行，但仍以实际任务数为上界，避免空开线程。"""
    if job_count < 1:
        return 1
    if configured:
        return min(configured, job_count)
    return min(4, os.cpu_count() or 1, job_count)


def _is_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda oom" in message or "cudamalloc" in message


class PageRuntimePool:
    """一组独占 PageRuntime，每个运行时同一时刻只处理一页。

    自动模式先完整加载一个运行时，再用加载前后的空闲显存差估计静态占用。
    后续运行时逐个创建，每次都重新查询；估计值之外还保留 gpu_free_margin_mb，
    给推理激活、KV cache 和显存分配器碎片留空间。该策略控制启动准入，不在
    推理过程中动态增删模型。
    """

    def __init__(
        self,
        root: str | Path,
        config: ParserConfig,
        *,
        runtime_factory: Callable[[], Any] | None = None,
        memory_probe: Callable[[], GpuMemorySnapshot | None] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.config = config
        self._runtime_factory = runtime_factory or self._default_factory
        self._memory_probe = memory_probe or gpu_memory_snapshot
        self._runtimes: list[Any] = []
        self._available: queue.Queue = queue.Queue()
        self._executor: ThreadPoolExecutor | None = None
        self._state_lock = threading.Lock()
        self._closed = False
        self.report: dict[str, Any] = {
            "mode": "fixed" if config.gpu_workers else "auto",
            "requested_workers": config.gpu_workers,
            "worker_count": 0,
            "max_auto_workers": config.max_gpu_workers,
            "free_margin_mb": config.gpu_free_margin_mb,
            "worker_floor_mb": config.gpu_worker_floor_mb,
            "events": [],
        }
        try:
            self._initialize()
            self._available = queue.Queue()
            for runtime in self._runtimes:
                self._available.put(runtime)
            self._executor = ThreadPoolExecutor(
                max_workers=len(self._runtimes), thread_name_prefix="ocr-page",
            )
            self.report["worker_count"] = len(self._runtimes)
        except BaseException:
            try:
                self._close_runtimes()
            except Exception:
                # 不掩盖真正的模型加载错误；关闭异常可由 worker 日志继续诊断。
                pass
            raise

    def _default_factory(self) -> Any:
        from .model_backend import PageRuntime
        return PageRuntime(self.root, self.config)

    @staticmethod
    def _snapshot_dict(snapshot: GpuMemorySnapshot | None) -> dict[str, Any] | None:
        return asdict(snapshot) if snapshot is not None else None

    def _create_runtime(self, index: int) -> Any:
        print(f"[runtime] loading worker {index}", file=sys.stderr, flush=True)
        runtime = self._runtime_factory()
        self._runtimes.append(runtime)
        self.report["events"].append({"event": "runtime_loaded", "index": index})
        print(f"[runtime] worker {index} loaded", file=sys.stderr, flush=True)
        return runtime

    def _initialize(self) -> None:
        requested = self.config.gpu_workers
        if requested:
            for index in range(requested):
                self._create_runtime(index)
            self.report["stop_reason"] = "fixed_worker_count"
            return

        before = self._memory_probe()
        self.report["events"].append({"event": "before_first", "memory": self._snapshot_dict(before)})
        self._create_runtime(0)
        after = self._memory_probe()
        self.report["events"].append({"event": "after_first", "memory": self._snapshot_dict(after)})
        if before is None or after is None:
            self.report["stop_reason"] = "gpu_memory_probe_unavailable"
            return

        observed = max(0, before.free_mb - after.free_mb)
        estimate = max(self.config.gpu_worker_floor_mb, observed)
        self.report["estimated_worker_mb"] = estimate
        while len(self._runtimes) < self.config.max_gpu_workers:
            current = self._memory_probe()
            if current is None:
                self.report["stop_reason"] = "gpu_memory_probe_lost"
                break
            required = estimate + self.config.gpu_free_margin_mb
            self.report["events"].append({
                "event": "admission_check", "next_index": len(self._runtimes),
                "free_mb": current.free_mb, "required_mb": required,
            })
            if current.free_mb < required:
                self.report["stop_reason"] = "insufficient_free_vram"
                break
            try:
                self._create_runtime(len(self._runtimes))
            except Exception as exc:
                # 自动扩张遇到 OOM 时保留已可用的运行时；其他错误仍需暴露。
                if not _is_oom(exc):
                    raise
                self.report["events"].append({"event": "oom_during_growth", "error": str(exc)})
                self.report["stop_reason"] = "oom_during_growth"
                break
            following = self._memory_probe()
            if following is not None:
                observed_now = max(0, current.free_mb - following.free_mb)
                estimate = max(estimate, observed_now)
                self.report["estimated_worker_mb"] = estimate
        else:
            self.report["stop_reason"] = "max_auto_workers_reached"

    def parse_pages(self, jobs: list[dict[str, Any]]) -> list[PageParseResult]:
        """并行执行页面任务，返回值顺序严格等于 jobs 顺序。"""
        if not jobs:
            return []
        with self._state_lock:
            if self._closed or self._executor is None:
                raise RuntimeError("Page runtime pool is closed")
            executor = self._executor

        def run(job: dict[str, Any]) -> PageParseResult:
            runtime = self._available.get()
            try:
                return runtime.parse_page(**job)
            finally:
                self._available.put(runtime)

        futures = [executor.submit(run, job) for job in jobs]
        return [future.result() for future in futures]

    def parse_page(self, **job: Any) -> PageParseResult:
        return self.parse_pages([job])[0]

    def parallelism_report(self) -> dict[str, Any]:
        return copy.deepcopy(self.report)

    def _close_runtimes(self) -> None:
        first_error = None
        for runtime in reversed(self._runtimes):
            try:
                runtime.close()
            except Exception as exc:  # 保证其余模型和 OCR 子进程仍能释放。
                if first_error is None:
                    first_error = exc
        self._runtimes.clear()
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=True)
        self._close_runtimes()

    def __enter__(self) -> "PageRuntimePool":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
