from services.inference.base import BaseInference
from transformers import AutoTokenizer, AutoModelForTextEncoding
from services.model_zoo import TEXT_ENCODER_MODELS
from typing import List, Dict
import torch
from loguru import logger
from services.helpers.text import prompt_clean

class T5TextEncoder(BaseInference):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        device_map: str | Dict[str, str] | None = None,
        max_length: int = 256,
        low_cpu_mem_usage: bool = False,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        super().__init__(model_name)
        self.model_config = TEXT_ENCODER_MODELS[model_name]
        self.device = device
        self.kwargs = kwargs
        self.max_length = max_length
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.device_map = device_map
        self.dtype = dtype
        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["_name_or_path"],
            cache_dir=self.model_config.get('cache_dir') or self.model_config.get('folder_path'),
        )
        self.model = AutoModelForTextEncoding.from_pretrained(
            self.model_config["folder_path"],
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            device_map=self.device_map,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        if self.device_map is None and self.device != "cpu":
            self.model.to(self.device)
        self.model.eval()

    # Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py#L140
    # Standard for T5 Tokenizer Models
    @torch.inference_mode()
    def encode(
        self,
        prompt: str | List[str],
        max_length: int = None,
        num_videos_per_prompt: int = 1,
        **kwargs,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)
        max_seq_length = max_length or self.max_length
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.model(
            text_input_ids.to(self.device), mask.to(self.device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
 
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_videos_per_prompt, seq_len, -1
        )

        return prompt_embeds
