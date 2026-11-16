import io
from contextlib import ExitStack
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import uuid
import zipfile

from fastapi.testclient import TestClient
from filelock import Timeout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from parser_service import JobStore, QueueFull, Settings, Supervisor, worker_main
from serve_api import create_app

TOKEN = "test-token-not-for-deployment"
AUTH = {"Authorization": "Bearer " + TOKEN}


class ManualSupervisor:
    def __init__(self, settings, store):
        self.ready = True
        self.state = "ready"

    def start(self):
        pass

    def close(self):
        pass

    def is_ready(self):
        return self.ready


def waiting_worker(settings, stop, events):
    if os.name != "nt":
        os.setsid()
    events.put({"event": "ready"})
    JobStore(settings.root).claim()
    stop.wait(30)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = Settings(self.root, TOKEN, max_upload_bytes=1024, max_pending=1)
        self.app = create_app(self.settings, ManualSupervisor)
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        self.client = contexts.enter_context(TestClient(self.app))
        self.store = self.app.state.store

    def submit(self, filename="test.pdf", data=b"%PDF-test"):
        return self.client.post("/v1/jobs", headers=AUTH, files={"file": (filename, data)})

    def test_authentication_health_and_readiness(self):
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        self.assertEqual(self.client.post("/v1/jobs", files={"file": ("x.pdf", b"x")}).status_code, 401)
        self.app.state.supervisor.ready = False
        self.assertEqual(self.client.get("/readyz").status_code, 503)
        self.assertEqual(self.submit().status_code, 503)

    def test_queue_status_and_capacity(self):
        response = self.submit()
        self.assertEqual(response.status_code, 202)
        job = response.json()
        self.assertEqual(job["status"], "queued")
        self.assertEqual(response.headers["location"], job["status_url"])
        self.assertEqual(self.client.get(job["status_url"], headers=AUTH).json()["job_id"], job["job_id"])
        self.assertEqual(self.submit().status_code, 429)
        self.assertEqual(len(list((self.root / "jobs").iterdir())), 1)
        self.assertEqual(self.client.get(job["result_url"], headers=AUTH).status_code, 409)

    def test_invalid_type_empty_and_oversize_files_leave_no_jobs(self):
        self.assertEqual(self.submit("test.exe").status_code, 415)
        self.assertEqual(self.submit(data=b"").status_code, 400)
        self.assertEqual(self.submit(data=b"x" * 1025).status_code, 413)
        self.assertEqual(list((self.root / "jobs").iterdir()), [])
        self.assertIsNone(self.store.claim())

    def test_multi_format_upload_allowlist(self):
        for filename in ("document.docx", "slides.pptx", "book.xlsx", "notes.txt",
                         "data.json", "bundle.zip", "bundle.tar", "scan.tiff"):
            with self.subTest(filename=filename):
                response = self.submit(filename, b"fixture")
                self.assertEqual(response.status_code, 202)
                self.store.claim()

    def test_chunked_and_declared_body_limits(self):
        response = self.client.post("/v1/jobs", headers={**AUTH, "Content-Length": str(2 * 1024 * 1024)}, content=b"x")
        self.assertEqual(response.status_code, 413)
        boundary = "test-boundary"
        def chunks():
            yield (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="x.pdf"\r\n\r\n').encode()
            yield b"x" * (2 * 1024 * 1024)
            yield f'\r\n--{boundary}--\r\n'.encode()
        response = self.client.post("/v1/jobs", headers={**AUTH, "Content-Type": f"multipart/form-data; boundary={boundary}"}, content=chunks())
        self.assertEqual(response.status_code, 413)
        self.assertIsNone(self.store.claim())

    def test_client_filename_cannot_choose_output_path(self):
        job = self.submit("../../outside.pdf").json()
        self.assertEqual(job["filename"], "outside.pdf")
        self.assertTrue((self.root / "jobs" / job["job_id"] / "input.pdf").is_file())
        self.assertFalse((self.root / "outside.pdf").exists())
        self.assertEqual(self.client.get("/v1/jobs/not-a-job", headers=AUTH).status_code, 404)

    def test_result_and_zip_contract_and_auth(self):
        job = self.submit().json()
        directory = self.root / "jobs" / job["job_id"]
        output = directory / "output"
        output.mkdir()
        (output / "manifest.json").write_text(json.dumps({"mode": "full_pdf", "status": "success"}))
        (output / "document.json").write_text(json.dumps({"page_count": 2, "routes": []}))
        (output / "document.md").write_text("recognized text")
        with zipfile.ZipFile(directory / "result.zip", "w") as archive:
            archive.write(output / "document.md", "document.md")
        self.store.update(job["job_id"], status="succeeded", completed_pages=2, page_count=2)
        result = self.client.get(job["result_url"], headers=AUTH)
        self.assertEqual(result.json()["document"]["page_count"], 2)
        self.assertEqual(result.json()["markdown"], "recognized text")
        self.assertEqual(self.client.get(job["download_url"]).status_code, 401)
        response = self.client.get(job["download_url"], headers=AUTH)
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(archive.read("document.md"), b"recognized text")

    def test_orchestrated_result_reads_nested_document_outputs(self):
        job = self.submit("document.docx", b"fixture").json()
        directory = self.root / "jobs" / job["job_id"]
        output = directory / "output"
        results = output / "results"
        results.mkdir(parents=True)
        (output / "manifest.json").write_text(json.dumps({"detected_format": "docx", "source_sha256": "abc"}))
        (results / "manifest.json").write_text(json.dumps({"mode": "multi_format_orchestration", "status": "success"}))
        (results / "document.json").write_text(json.dumps({"pages": [], "routes": []}))
        (results / "document.md").write_text("office text")
        self.store.update(job["job_id"], status="succeeded")
        response = self.client.get(job["result_url"], headers=AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["markdown"], "office text")
        self.assertEqual(response.json()["manifest"]["preparation"]["detected_format"], "docx")

    def test_restart_marks_running_failed_but_preserves_queue(self):
        first = self.submit().json()["job_id"]
        self.store.claim()
        second = self.submit().json()["job_id"]
        JobStore(self.root).fail_running("service_restarted", "interrupted")
        self.assertEqual(self.store.get(first)["status"], "failed")
        self.assertEqual(self.store.claim()["id"], second)

    def test_queue_admission_is_atomic(self):
        successes = []
        def enqueue():
            try:
                self.store.enqueue(uuid.uuid4().hex, "test.pdf", 1)
                successes.append(1)
            except QueueFull:
                pass
        threads = [threading.Thread(target=enqueue) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(successes), 1)

    def test_missing_token_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "TOKEN"):
            create_app(Settings(self.root, ""), ManualSupervisor)


class ResidentWorkerTests(unittest.TestCase):
    def test_two_documents_share_one_runtime_and_reset_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = JobStore(root)
            ids = [uuid.uuid4().hex for _ in range(2)]
            for job_id in ids:
                directory = root / "jobs" / job_id
                directory.mkdir(parents=True)
                (directory / "input.pdf").write_bytes(b"fixture")
                store.enqueue(job_id, "file.pdf", 4)
            stop = threading.Event()
            events = __import__("queue").Queue()
            calls, closed = [], []

            class Runtime:
                def __init__(self):
                    calls.append("init")

                def close(self):
                    closed.append(True)

            def orchestrate(source, output, *, progress, **kwargs):
                output.mkdir()
                (output / "document.md").write_text(source.parent.name)
                (output / "manifest.json").write_text(json.dumps({"mode": "multi_format_orchestration"}))
                (output / "document.json").write_text("{}")
                progress(2)
                calls.append(source.parent.name)
                if len(calls) == 3:
                    stop.set()
                return {"page_count": 2}

            with patch("run_single_page.ParserRuntime", Runtime), patch("run_document_orchestrator.run", orchestrate), patch("parser_service.inspect_input", return_value=("pdf", 2)), patch("parser_service.os.setsid", create=True):
                worker_main(Settings(root, TOKEN), stop, events)
            self.assertEqual(calls, ["init", *ids])
            self.assertEqual(closed, [True])
            for job_id in ids:
                self.assertEqual(store.get(job_id)["status"], "succeeded")
                self.assertEqual(store.get(job_id)["completed_pages"], 2)
                with zipfile.ZipFile(root / "jobs" / job_id / "result.zip") as archive:
                    self.assertEqual(archive.read("document.md").decode(), job_id)

    def test_supervisor_times_out_job_and_excludes_second_service(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = Settings(Path(temp), TOKEN, job_timeout=1, startup_timeout=10)
            store = JobStore(settings.root)
            job_id = uuid.uuid4().hex
            store.enqueue(job_id, "test.pdf", 1)
            supervisor = Supervisor(settings, store, waiting_worker)
            supervisor.start()
            try:
                other = Supervisor(settings, store, waiting_worker)
                with self.assertRaises(Timeout):
                    other.start()
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline and store.get(job_id)["status"] != "failed":
                    time.sleep(0.05)
                self.assertEqual(store.get(job_id)["error_code"], "job_timeout")
            finally:
                supervisor.close()
            self.assertFalse(supervisor.process.is_alive())


if __name__ == "__main__":
    unittest.main()
