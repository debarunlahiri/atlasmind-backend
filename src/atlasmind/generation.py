from collections.abc import Iterator
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from threading import Thread
from typing import Any, Optional

import torch
from PIL import Image

DEFAULT_MULTIMODAL_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def resolve_compute_device(device_name: str) -> torch.device:
    """Resolve local compute, preferring Apple's Metal Performance Shaders."""
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "Apple Metal acceleration was requested, but PyTorch MPS is unavailable. "
                "Use an MPS-enabled PyTorch installation on Apple Silicon or set "
                "ATLASMIND_COMPUTE_DEVICE=cpu."
            )
        return torch.device("mps")
    if device_name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device_name == "cpu":
        return torch.device("cpu")
    raise ValueError("Compute device must be 'mps', 'cpu', or 'auto'")


@dataclass(frozen=True)
class GeneratedText:
    text: str
    model: str


class LocalMultimodalGenerator:
    """Generate text from text-only or text-and-image prompts on local hardware."""

    def __init__(
        self,
        model_id: str = DEFAULT_MULTIMODAL_MODEL,
        cache_dir: Optional[Path] = None,
        device_name: str = "mps",
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_id
        self.cache_dir = cache_dir
        self.device = resolve_compute_device(device_name)
        self.local_files_only = local_files_only
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            if find_spec("torchvision") is None:
                raise RuntimeError(
                    "The multimodal model requires torchvision, but it is not installed in "
                    "AtlasMind's "
                    "virtual environment. Run '.venv/bin/python -m pip install -r "
                    "requirements.txt' and restart the API server."
                )
            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    "AtlasMind could not import the multimodal Transformers components. "
                    "Install the declared torch, torchvision, transformers, and pillow "
                    "versions with '.venv/bin/python -m pip install -r requirements.txt', "
                    "then restart the API server."
                ) from error

            cache_dir = str(self.cache_dir) if self.cache_dir else None
            dtype = torch.float16 if self.device.type == "mps" else torch.float32
            common_options = {
                "cache_dir": cache_dir,
                "local_files_only": self.local_files_only,
            }
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    **common_options,
                )
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                    attn_implementation="eager",
                    **common_options,
                ).to(self.device)
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    "AtlasMind could not load the configured multimodal model or processor. "
                    "Install all packages "
                    "from requirements.txt and restart the API server. "
                    f"Underlying error: {error}"
                ) from error
            self._model.eval()
        return self._model, self._processor

    def _generation_inputs(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        model, processor = self._load()
        content: list[dict[str, object]] = []
        if image is not None:
            content.append({"type": "image", "image": image.convert("RGB")})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            name: (
                value.to(self.device, dtype=model.dtype)
                if torch.is_floating_point(value)
                else value.to(self.device)
            )
            for name, value in inputs.items()
        }
        return model, processor, inputs

    @staticmethod
    def _generation_options(max_new_tokens: int, temperature: float) -> dict[str, object]:
        generation_options: dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_options["temperature"] = temperature
        return generation_options

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        image: Optional[Image.Image] = None,
    ) -> GeneratedText:
        model, processor, inputs = self._generation_inputs(prompt, image)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                **self._generation_options(max_new_tokens, temperature),
            )

        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = output_ids[:, prompt_length:]
        text = processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
        )[0].strip()
        if not text:
            raise RuntimeError("The local multimodal model returned no text")
        return GeneratedText(text=text, model=self.model_name)

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        image: Optional[Image.Image] = None,
    ) -> Iterator[str]:
        """Yield decoded text chunks while the local model is generating."""
        from transformers import TextIteratorStreamer

        model, processor, inputs = self._generation_inputs(prompt, image)
        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        errors: list[Exception] = []

        def generate_in_background() -> None:
            try:
                with torch.inference_mode():
                    model.generate(
                        **inputs,
                        **self._generation_options(max_new_tokens, temperature),
                        streamer=streamer,
                    )
            except Exception as error:
                errors.append(error)
                streamer.end()

        worker = Thread(target=generate_in_background, daemon=True)
        worker.start()
        for text in streamer:
            if text:
                yield text
        worker.join()
        if errors:
            raise RuntimeError(f"Local model streaming failed: {errors[0]}") from errors[0]
