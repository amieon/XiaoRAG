from __future__ import annotations

import contextlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chapter_tree.heading_classifier import HeadingExample
from chapter_tree.heading_hierarchy import HeadingHierarchyResolver
from chapter_tree.local_hf_classifier import (
    LocalHFHeadingClassifier, _constrained_candidates, _prefix_allowed_tokens,
)


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self, response_factory):
        self.response_factory = response_factory
        self.template_options = []

    def apply_chat_template(self, messages, **options):
        self.template_options.append(options)
        self.payload = {"messages": messages}
        return "rendered-chat-prompt"

    def __call__(self, texts, return_tensors=None, add_special_tokens=True):
        if return_tensors == "pt":
            assert texts == ["rendered-chat-prompt"]
            return FakeInputs(input_ids=[[10, 11, 12]])
        assert not add_special_tokens
        return {"input_ids": [20, 21]}

    def decode(self, tokens, skip_special_tokens):
        assert list(tokens) == [20, 21]
        assert skip_special_tokens
        return self.response_factory(self.payload)


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.generate_options = []

    def eval(self):
        self.evaluated = True

    def generate(self, **options):
        self.generate_options.append(options)
        return [[10, 11, 12, 20, 21]]


class FakeTorch:
    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


def example(position: int, text: str) -> HeadingExample:
    return HeadingExample(
        block_id=f"doc_p0001_b{position:04d}", page_id="doc_p0001", document_id="doc",
        text=text, context={}, raw_label="text", reading_order=position,
        source_page_result="synthetic",
    )


class LocalHFClassifierTests(unittest.TestCase):
    def test_local_directory_drives_classification_and_hierarchy_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "Qwen-Quantized"
            model_path.mkdir()
            (model_path / "config.json").write_text("{}", encoding="utf-8")
            model = FakeModel()
            model_calls = []
            tokenizer_calls = []

            def response_factory(payload):
                user = json.loads(payload["messages"][1]["content"])
                if "blocks" in user:
                    return json.dumps({"predictions": [{
                        "block_id": item["block_id"], "label": "heading", "heading_level": 6,
                    } for item in user["blocks"]]})
                return json.dumps({"parent_id": "root"})

            def tokenizer_loader(path, **options):
                tokenizer_calls.append((path, options))
                return FakeTokenizer(response_factory)

            def model_loader(path, **options):
                model_calls.append((path, options))
                return model

            classifier = LocalHFHeadingClassifier(
                model_path,
                max_tokens=4096,
                disable_thinking=True,
                model_loader=model_loader,
                tokenizer_loader=tokenizer_loader,
                torch_module=FakeTorch(),
            )
            rows = classifier.predict([example(0, "无编号标题")])
            self.assertEqual(rows[0]["label"], "heading")
            self.assertEqual(rows[0]["heading_level"], 6)
            self.assertEqual(rows[0]["model"], "Qwen-Quantized")
            self.assertEqual(model_calls, [(str(model_path), {
                "device_map": "auto", "torch_dtype": "auto", "local_files_only": True,
            })])
            self.assertEqual(tokenizer_calls, [(str(model_path), {"local_files_only": True})])
            self.assertFalse(model.generate_options[0]["do_sample"])
            self.assertEqual(model.generate_options[0]["max_new_tokens"], 3)
            self.assertTrue(callable(model.generate_options[0]["prefix_allowed_tokens_fn"]))
            self.assertEqual(classifier.tokenizer.template_options[0]["enable_thinking"], False)

            items = [example(0, "总论"), example(1, "子节")]
            predictions = [{"block_id": item.block_id, "label": "heading", "heading_level": 1} for item in items]
            resolved = HeadingHierarchyResolver(classifier, mode="streaming").resolve(items, predictions)
            self.assertEqual([row["heading_level"] for row in resolved], [1, 1])
            self.assertEqual(resolved[1]["hierarchy_source"], "streaming_model")

    def test_candidate_sets_and_token_prefixes_only_allow_complete_legal_outputs(self):
        classification = _constrained_candidates({"messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": json.dumps({"blocks": [{"block_id": "b0"}]})},
        ]})
        self.assertEqual(len(classification), 7)
        decoded = [json.loads(item)["predictions"][0] for item in classification]
        self.assertEqual({item["label"] for item in decoded}, {"body", "heading"})
        self.assertEqual({item["heading_level"] for item in decoded}, {None, 1, 2, 3, 4, 5, 6})

        hierarchy = _constrained_candidates({"messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": json.dumps({
                "current": {"text": "子节"},
                "active_path": [{"heading_id": "root"}, {"heading_id": "h000"}],
            })},
        ]})
        self.assertEqual([json.loads(item)["parent_id"] for item in hierarchy], ["root", "h000"])

        class Tokens(list):
            def tolist(self):
                return list(self)

        allowed = _prefix_allowed_tokens([(1, 2), (1, 3), (4,)], 1, 9)
        self.assertEqual(allowed(0, Tokens([99])), [1, 4])
        self.assertEqual(allowed(0, Tokens([99, 1])), [2, 3])
        self.assertEqual(allowed(0, Tokens([99, 1, 2])), [9])
        self.assertEqual(allowed(0, Tokens([99, 4])), [9])

    def test_explicit_cuda_fails_immediately_when_torch_has_no_cuda(self):
        class UnavailableCuda:
            @staticmethod
            def is_available():
                return False

        class NoCudaTorch(FakeTorch):
            cuda = UnavailableCuda()

        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "Qwen"
            model_path.mkdir()
            (model_path / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "torch.cuda.is_available"):
                LocalHFHeadingClassifier(
                    model_path, device="cuda:0",
                    model_loader=lambda *args, **kwargs: FakeModel(),
                    tokenizer_loader=lambda *args, **kwargs: FakeTokenizer(lambda payload: ""),
                    torch_module=NoCudaTorch(),
                )

    def test_local_model_requires_real_directory_and_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "containing config.json"):
                LocalHFHeadingClassifier(Path(temporary) / "missing")


if __name__ == "__main__":
    unittest.main()
