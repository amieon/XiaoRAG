"""常驻轻量 PaddleOCR 进程，通过 stdin/stdout 的 JSON Lines 接收裁图路径。"""
from __future__ import annotations

import contextlib
import faulthandler
import json
import os
from pathlib import Path
import sys
import time
import traceback


def _json_value(value):
    if isinstance(value, dict):
        return value
    payload = getattr(value, "json", None)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        return json.loads(payload)
    if isinstance(payload, dict):
        return payload
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _as_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def _extract_result(result):
    data = _json_value(result)
    texts = _as_list(data.get("rec_texts", data.get("text", [])))
    scores = _as_list(data.get("rec_scores", data.get("scores", [])))
    boxes = _as_list(data.get("rec_boxes", data.get("dt_polys", [])))
    return {
        "result": "\n".join(str(text) for text in texts),
        "line_scores": [float(score) for score in scores],
        "line_boxes": boxes,
    }


def _configure_environment(project: Path):
    # 与父进程保持一致，且必须先于 paddleocr/paddle 导入。
    os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(project / ".cache" / "paddlex"))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("HF_HOME", str(project / ".cache" / "huggingface-paddleocr"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def build_model(project: Path):
    from paddleocr import PaddleOCR

    det_name = os.environ.get("PADDLEOCR_TEXT_DET_MODEL_NAME", "PP-OCRv6_small_det")
    rec_name = os.environ.get("PADDLEOCR_TEXT_REC_MODEL_NAME", "PP-OCRv6_small_rec")
    kwargs = {
        "lang": os.environ.get("PADDLEOCR_LANG", "ch"),
        "device": os.environ.get("PADDLEOCR_DEVICE", "gpu:0"),
        "engine": os.environ.get("PADDLEOCR_ENGINE", "paddle_static"),
        "text_detection_model_name": det_name,
        "text_recognition_model_name": rec_name,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "text_recognition_batch_size": int(os.environ.get("PADDLEOCR_REC_BATCH", "6")),
    }
    # 目录存在时使用本地权重；否则交给 PaddleOCR 的公开模型解析逻辑。
    det_dir = Path(os.environ.get("PADDLEOCR_TEXT_DET_MODEL_DIR", Path("models") / det_name))
    rec_dir = Path(os.environ.get("PADDLEOCR_TEXT_REC_MODEL_DIR", Path("models") / rec_name))
    if det_dir.is_dir():
        kwargs["text_detection_model_dir"] = str(det_dir)
    if rec_dir.is_dir():
        kwargs["text_recognition_model_dir"] = str(rec_dir)
    return PaddleOCR(**kwargs)


def main():
    project = Path(__file__).resolve().parents[1]
    _configure_environment(project)
    import numpy as np

    protocol = sys.stdout
    # Windows 默认控制台编码可能是 GBK；JSON Lines 协议必须始终是 UTF-8。
    # 原生库仍可能绕过此文本包装器写 fd 1，父进程会把这类非 JSON 字节当诊断日志。
    protocol.reconfigure(encoding="utf-8", errors="strict", newline="\n")
    sys.stdin.reconfigure(encoding="utf-8", errors="strict")
    # 模型库可能直接 print；协议消息只能走原 stdout，其余日志统一转 stderr。
    with contextlib.redirect_stdout(sys.stderr):
        started = time.perf_counter()
        engine = os.environ.get("PADDLEOCR_ENGINE", "paddle_static")
        device = os.environ.get("PADDLEOCR_DEVICE", "gpu:0")
        print(f"[light-ocr] initializing engine={engine} device={device}", file=sys.stderr, flush=True)
        startup_timeout = float(os.environ.get("PADDLEOCR_WORKER_READY_TIMEOUT", "600"))
        # 真正死锁时留下 worker 的 Python 栈并退出；父进程随后会收到 EOF，而非无声悬挂。
        faulthandler.dump_traceback_later(startup_timeout, exit=True)
        try:
            model = build_model(project)
        finally:
            faulthandler.cancel_dump_traceback_later()
        print(
            f"[light-ocr] model initialized in {time.perf_counter() - started:.2f}s; sending READY",
            file=sys.stderr,
            flush=True,
        )
        ready = {
            "event": "READY",
            "engine": "paddleocr",
            "inference_engine": engine,
            "det_model": os.environ.get("PADDLEOCR_TEXT_DET_MODEL_NAME", "PP-OCRv6_small_det"),
            "rec_model": os.environ.get("PADDLEOCR_TEXT_REC_MODEL_NAME", "PP-OCRv6_small_rec"),
            "load_seconds": time.perf_counter() - started,
            "lang": os.environ.get("PADDLEOCR_LANG", "ch"),
            "device": os.environ.get("PADDLEOCR_DEVICE", "gpu:0"),
            "rec_batch_num": int(os.environ.get("PADDLEOCR_REC_BATCH", "6")),
        }
    print(json.dumps(ready, ensure_ascii=False), file=protocol, flush=True)

    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            break
        try:
            outputs = []
            started = time.perf_counter()
            print(f"[light-ocr] request started: crops={len(request['images'])}", file=sys.stderr, flush=True)
            with contextlib.redirect_stdout(sys.stderr):
                for name in request["images"]:
                    image = np.load(name, allow_pickle=False)
                    result = next(iter(model.predict(image)), None)
                    outputs.append(_extract_result(result) if result is not None else {
                        "result": "", "line_scores": [], "line_boxes": [],
                    })
            elapsed = time.perf_counter() - started
            print(f"[light-ocr] request finished in {elapsed:.2f}s", file=sys.stderr, flush=True)
            print(json.dumps({
                "outputs": outputs, "seconds": elapsed,
            }, ensure_ascii=False), file=protocol, flush=True)
        except Exception:
            # 单次请求失败通过协议返回，不让 worker 静默退出后造成父进程悬挂。
            print(json.dumps({"error": traceback.format_exc()}, ensure_ascii=False), file=protocol, flush=True)


if __name__ == "__main__":
    main()
