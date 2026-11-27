"""直接加载 Hugging Face 本地模型目录的零样本标题分类器。

支持 Transformers 能加载的原始或量化 CausalLM；量化方式由模型 config 决定。
不启动 Ollama／vLLM，不访问网络，标题识别与层级恢复共用同一模型实例。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .zero_shot_classifier import OpenAICompatibleHeadingClassifier


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
        model_loader: Callable[..., Any] | None = None,
        tokenizer_loader: Callable[..., Any] | None = None,
        torch_module: Any = None,
    ) -> None:
        path = Path(model_path).resolve()
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError(f"Local HF model must be a directory containing config.json: {path}")
        if max_tokens < 256:
            raise ValueError("max_tokens must be at least 256")
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

        self.model_path = path
        self.model = path.name
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        # 仅保留统一请求结构；本地 Transformers 不声称具有服务端 JSON grammar 约束。
        self.use_response_format = use_response_format
        self.torch = torch_module
        self.tokenizer = tokenizer_loader(str(path), local_files_only=True)
        self.hf_model = model_loader(
            str(path), device_map=device, torch_dtype="auto", local_files_only=True,
        )
        self.hf_model.eval()

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
        with self.torch.inference_mode():
            generated = self.hf_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0][input_length:]
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "length" if len(new_tokens) >= max_tokens else "stop",
            }],
            "usage": {"completion_tokens": len(new_tokens)},
        }
