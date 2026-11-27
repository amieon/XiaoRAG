from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from doc_parser.model_backend import PaddleModelBackend, _read_response
from doc_parser.pipeline import ParserConfig


class NonClosingBytesIO(io.BytesIO):
    def close(self):
        # 测试结束后仍需审计父进程写入的原始请求字节。
        pass


class FakeWorker:
    def __init__(self, response: bytes):
        self.stdout = io.BytesIO(response)
        self.stdin = NonClosingBytesIO()
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


class ModelBackendEncodingTests(unittest.TestCase):
    def test_gbk_native_log_before_utf8_ready_does_not_crash(self):
        # 中文 Windows 原生库可能绕过 Python redirect_stdout，直接写 GBK 字节。
        log = "Paddle 原生库：正在初始化中文模型".encode("gb18030")
        ready = json.dumps({"event": "READY", "device": "中文 GPU"}, ensure_ascii=False).encode("utf-8")
        errors = io.StringIO()
        with redirect_stderr(errors):
            response = _read_response(io.BytesIO(log + b"\n" + ready + b"\n"), timeout=2)
        self.assertEqual(response["event"], "READY")
        self.assertEqual(response["device"], "中文 GPU")
        self.assertIn("正在初始化中文模型", errors.getvalue())

    def test_non_utf8_json_fails_instead_of_waiting_for_timeout(self):
        response = json.dumps({"outputs": [{"result": "中文正文"}]}, ensure_ascii=False).encode("gb18030")
        with self.assertRaisesRegex(RuntimeError, "JSON protocol is not valid UTF-8"):
            _read_response(io.BytesIO(response + b"\n"), timeout=2)

    def test_worker_process_uses_binary_pipes_and_utf8_environment(self):
        worker = FakeWorker(b'{"event":"READY"}\n')
        fake_paddlex = SimpleNamespace(create_model=lambda **kwargs: object())
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict(sys.modules, {"paddlex": fake_paddlex}), \
                patch("doc_parser.model_backend.subprocess.Popen", return_value=worker) as popen:
            backend = PaddleModelBackend(temp, ParserConfig())
            kwargs = popen.call_args.kwargs
            self.assertNotIn("text", kwargs)
            self.assertNotIn("encoding", kwargs)
            self.assertEqual(kwargs["env"]["PYTHONIOENCODING"], "utf-8")
            backend.close()
        self.assertEqual(worker.stdin.getvalue(), b'{"stop":true}\n')

    def test_recognize_text_writes_utf8_request_and_reads_chinese_response(self):
        response = json.dumps({
            "outputs": [{"result": "中文识别结果", "line_scores": [0.9], "line_boxes": []}],
        }, ensure_ascii=False).encode("utf-8") + b"\n"
        worker = FakeWorker(response)
        backend = PaddleModelBackend.__new__(PaddleModelBackend)
        backend.worker = worker
        with patch.dict("os.environ", {"PADDLEOCR_WORKER_INFER_TIMEOUT": "2"}):
            result = backend.recognize_text([np.zeros((2, 2), dtype=np.uint8)])
        self.assertEqual(result[0]["result"], "中文识别结果")
        request = json.loads(worker.stdin.getvalue().decode("utf-8"))
        self.assertEqual(len(request["images"]), 1)


if __name__ == "__main__":
    unittest.main()
