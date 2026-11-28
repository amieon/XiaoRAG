"""直接加载 Hugging Face 本地模型目录的零样本标题分类器。

支持 Transformers 能加载的原始或量化 CausalLM；量化方式由模型 config 决定。
不启动 Ollama／vLLM，不访问网络，标题识别与层级恢复共用同一模型实例。
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

from .zero_shot_classifier import OpenAICompatibleHeadingClassifier


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _constrained_candidates(payload: dict[str, Any]) -> list[str]:
    try:
        user = json.loads(payload["messages"][1]["content"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return []
    blocks = user.get("blocks") if isinstance(user, dict) else None
    if isinstance(blocks, list) and len(blocks) == 1:
        block_id = blocks[0].get("block_id")
        if isinstance(block_id, str):
            return [
                _compact_json({"predictions": [{
                    "block_id": block_id, "label": "body", "heading_level": None,
                }]}),
                *[
                    _compact_json({"predictions": [{
                        "block_id": block_id, "label": "heading", "heading_level": level,
                    }]})
                    for level in range(1, 7)
                ],
            ]
    active_path = user.get("active_path") if isinstance(user, dict) else None
    if isinstance(active_path, list):
        parent_ids = []
        for item in active_path:
            if isinstance(item, dict) and isinstance(item.get("heading_id"), str):
                parent_ids.append(item["heading_id"])
        if parent_ids and len(parent_ids) == len(set(parent_ids)):
            return [_compact_json({"parent_id": parent_id}) for parent_id in parent_ids]
    return []


def _prefix_allowed_tokens(
    sequences: list[tuple[int, ...]], prompt_length: int, eos_token_id: int,
) -> Callable[[int, Any], list[int]]:
    def allowed(batch_id: int, input_ids: Any) -> list[int]:
        generated_prefix = input_ids[prompt_length:]
        values = generated_prefix.tolist() if hasattr(generated_prefix, "tolist") else generated_prefix
        prefix = tuple(values)
        next_tokens = set()
        for sequence in sequences:
            if len(prefix) < len(sequence) and sequence[:len(prefix)] == prefix:
                next_tokens.add(sequence[len(prefix)])
            elif prefix == sequence:
                next_tokens.add(eos_token_id)
        # 正常不会发生；保留 EOS 让生成失败为空并由现有 JSON 校验拒绝。
        return sorted(next_tokens) or [eos_token_id]
    return allowed


class LocalHFHeadingClassifier(OpenAICompatibleHeadingClassifier):
    """复用远程分类器的提示词和严格校验，仅替换模型调用层。"""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        max_tokens: int = 4096,
        disable_thinking: bool = False,
        use_response_format: bool = True,
        max_retries: int = 2,
        model_loader: Callable[..., Any] | None = None,
        tokenizer_loader: Callable[..., Any] | None = None,
        torch_module: Any = None,
    ) -> None:
        path = Path(model_path).resolve()
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError(f"Local HF model must be a directory containing config.json: {path}")
        if max_tokens < 256:
            raise ValueError("max_tokens must be at least 256")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if device != "auto" and not device.strip():
            raise ValueError("device must be auto or a non-empty device specification")

        if model_loader is None or tokenizer_loader is None or torch_module is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Local HF mode requires torch, transformers and accelerate; "
                    "AWQ/GPTQ models also require their matching quantization runtime"
                ) from exc
            model_loader = model_loader or AutoModelForCausalLM.from_pretrained
            tokenizer_loader = tokenizer_loader or AutoTokenizer.from_pretrained
            torch_module = torch_module or torch

        requested_device = device.strip()
        if requested_device.startswith("cuda"):
            cuda = getattr(torch_module, "cuda", None)
            if cuda is None or not cuda.is_available():
                raise RuntimeError(
                    f"--device {requested_device} requires CUDA PyTorch, but torch.cuda.is_available() "
                    "is false. Use a CUDA-enabled heading environment or explicitly choose CPU."
                )
        self.model_path = path
        self.model = path.name
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.disable_thinking = disable_thinking
        self.generation_count = 0
        # 仅保留统一请求结构；本地 Transformers 不声称具有服务端 JSON grammar 约束。
        self.use_response_format = use_response_format
        self.torch = torch_module
        print(f"[local-hf] loading tokenizer: {path}", file=sys.stderr, flush=True)
        self.tokenizer = tokenizer_loader(str(path), local_files_only=True)
        print(
            f"[local-hf] loading weights: requested_device={requested_device}",
            file=sys.stderr, flush=True,
        )
        self.hf_model = model_loader(
            str(path), device_map=requested_device, torch_dtype="auto", local_files_only=True,
        )
        self.hf_model.eval()
        actual_device = str(getattr(self.hf_model, "device", "unknown"))
        print(
            f"[local-hf] READY model={self.model}; actual_device={actual_device}",
            file=sys.stderr, flush=True,
        )
        if actual_device == "cpu" and requested_device == "auto":
            print(
                "[local-hf] WARNING: auto resolved to CPU; generation may be slow and "
                "nvidia-smi will not show this process",
                file=sys.stderr, flush=True,
            )

    def _predict_batch(self, examples):
        # 本地候选约束按单块定义；多块在程序侧拆开，不让模型自行发明块 ID。
        if len(examples) > 1:
            predictions = []
            for example in examples:
                predictions.extend(super()._predict_batch([example]))
            return predictions
        return super()._predict_batch(examples)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        template_options: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.disable_thinking:
            template_options["enable_thinking"] = False
        text = self.tokenizer.apply_chat_template(payload["messages"], **template_options)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.hf_model.device)
        input_length = len(inputs["input_ids"][0])
        max_tokens = int(payload["max_tokens"])
        candidates = _constrained_candidates(payload)
        generation_options: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if candidates:
            token_sequences = []
            for candidate in candidates:
                encoded = self.tokenizer(candidate, add_special_tokens=False)["input_ids"]
                if encoded and isinstance(encoded[0], list):
                    encoded = encoded[0]
                if not encoded:
                    raise RuntimeError("Tokenizer produced an empty constrained candidate")
                token_sequences.append(tuple(int(token) for token in encoded))
            generation_options["prefix_allowed_tokens_fn"] = _prefix_allowed_tokens(
                token_sequences, input_length, self.tokenizer.eos_token_id,
            )
            generation_options["max_new_tokens"] = min(
                max_tokens, max(map(len, token_sequences)) + 1,
            )
        context_limit = int(getattr(getattr(self.hf_model, "config", None), "max_position_embeddings", 0) or 0)
        required_tokens = input_length + generation_options["max_new_tokens"]
        if context_limit and required_tokens > context_limit:
            raise ValueError(
                "model_context_exceeded: "
                f"input_tokens={input_length}, max_new_tokens={generation_options['max_new_tokens']}, "
                f"context_limit={context_limit}; reflow oversized native IR blocks instead of truncating"
            )
        self.generation_count += 1
        generation_id = self.generation_count
        started = time.perf_counter()
        print(
            f"[local-hf] generation {generation_id} started: "
            f"device={self.hf_model.device}, input_tokens={input_length}, "
            f"max_new_tokens={generation_options['max_new_tokens']}, "
            f"constrained_candidates={len(candidates)}",
            file=sys.stderr, flush=True,
        )
        with self.torch.inference_mode():
            generated = self.hf_model.generate(**generation_options)
        new_tokens = generated[0][input_length:]
        elapsed = time.perf_counter() - started
        print(
            f"[local-hf] generation {generation_id} finished: "
            f"completion_tokens={len(new_tokens)}, elapsed_seconds={elapsed:.2f}",
            file=sys.stderr, flush=True,
        )
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {
            "choices": [{
                "message": {"content": content},
                "finish_reason": (
                    "length" if len(new_tokens) >= generation_options["max_new_tokens"] else "stop"
                ),
            }],
            "usage": {"completion_tokens": len(new_tokens)},
        }
