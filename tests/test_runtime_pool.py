from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_parser.pipeline import ParserConfig
from doc_parser.runtime_pool import GpuMemorySnapshot, PageRuntimePool, resolve_cpu_workers


def memory(free, total=48_000):
    return GpuMemorySnapshot(device_index=0, free_mb=free, total_mb=total)


class FakeRuntime:
    def __init__(self, index, owner):
        self.index = index
        self.owner = owner
        self.closed = False

    def parse_page(self, **job):
        with self.owner.lock:
            self.owner.active += 1
            self.owner.max_active = max(self.owner.max_active, self.owner.active)
        try:
            # 第一页故意最慢：若错误地按完成顺序收集，返回顺序就会被打乱。
            time.sleep(job.get("delay", 0))
            return {"page_no": job["page_no"], "runtime": self.index}
        finally:
            with self.owner.lock:
                self.owner.active -= 1

    def close(self):
        self.closed = True


class RuntimeOwner:
    def __init__(self):
        self.created = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def factory(self):
        runtime = FakeRuntime(len(self.created), self)
        self.created.append(runtime)
        return runtime


class RuntimePoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.owner = RuntimeOwner()

    def tearDown(self):
        self.temp.cleanup()

    def test_auto_growth_uses_observed_vram_and_rechecks_each_worker(self):
        values = iter([
            memory(40_000), memory(30_000),  # 第一个运行时实际增加约 10 GiB 占用
            memory(30_000), memory(20_000),  # 准入并加载第二个
            memory(20_000), memory(9_000),   # 准入并加载第三个
            memory(9_000),                   # 余量不足，不再加载第四个
        ])
        config = ParserConfig(gpu_free_margin_mb=4_000, max_gpu_workers=4)
        pool = PageRuntimePool(
            self.root, config, runtime_factory=self.owner.factory, memory_probe=lambda: next(values),
        )
        self.assertEqual(len(self.owner.created), 3)
        self.assertEqual(pool.report["estimated_worker_mb"], 11_000)
        self.assertEqual(pool.report["stop_reason"], "insufficient_free_vram")
        pool.close()
        self.assertTrue(all(runtime.closed for runtime in self.owner.created))

    def test_auto_stops_at_one_when_probe_is_unavailable(self):
        pool = PageRuntimePool(
            self.root, ParserConfig(), runtime_factory=self.owner.factory, memory_probe=lambda: None,
        )
        self.assertEqual(pool.report["worker_count"], 1)
        self.assertEqual(pool.report["stop_reason"], "gpu_memory_probe_unavailable")
        pool.close()

    def test_fixed_workers_do_not_require_memory_probe(self):
        def forbidden_probe():
            raise AssertionError("固定 worker 数不应依赖显存自动探测")
        pool = PageRuntimePool(
            self.root, ParserConfig(gpu_workers=2),
            runtime_factory=self.owner.factory, memory_probe=forbidden_probe,
        )
        self.assertEqual(pool.report["worker_count"], 2)
        pool.close()

    def test_pages_run_concurrently_but_return_in_job_order(self):
        pool = PageRuntimePool(
            self.root, ParserConfig(gpu_workers=2), runtime_factory=self.owner.factory,
        )
        jobs = [
            {"page_no": 1, "delay": 0.08},
            {"page_no": 2, "delay": 0.01},
            {"page_no": 3, "delay": 0.01},
            {"page_no": 4, "delay": 0.01},
        ]
        results = pool.parse_pages(jobs)
        self.assertEqual([item["page_no"] for item in results], [1, 2, 3, 4])
        self.assertGreaterEqual(self.owner.max_active, 2)
        pool.close()

    def test_cpu_auto_workers_are_bounded_by_jobs(self):
        self.assertEqual(resolve_cpu_workers(0, 1), 1)
        self.assertEqual(resolve_cpu_workers(8, 3), 3)
        self.assertEqual(resolve_cpu_workers(0, 100), min(4, __import__("os").cpu_count() or 1))


if __name__ == "__main__":
    unittest.main()
