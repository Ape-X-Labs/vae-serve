from services.inference.base import BaseInference
import diffusers
from services.model_zoo import VAE_MODELS
from typing import List, Dict
import torch
from tqdm.auto import tqdm

class VAE(BaseInference):
    def __init__(self, model_name: str, device: str = "cuda", dtype: torch.dtype = torch.float16, offload_to_cpu: bool = False, low_cpu_mem_usage: bool = False, device_map: str | Dict[str, str] | None = None, micro_batch_size: int = 4, **kwargs):
        super().__init__(model_name)
        self.device = device
        self.dtype = dtype
        self.offload_to_cpu = offload_to_cpu
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.device_map = device_map
        self.kwargs = kwargs
        self.model_config = VAE_MODELS[model_name]
        self.micro_batch_size = micro_batch_size
        self._load_model()

    def _load_model(self):
        class_name = self.model_config["_class_name"]
        self.model = getattr(diffusers, class_name).from_pretrained(
            self.model_config["folder_path"],
            torch_dtype=self.dtype,
            device_map=self.device_map,
            offload_to_cpu=self.offload_to_cpu,
            low_cpu_mem_usage=self.low_cpu_mem_usage
        )
        if self.device_map is None and self.device != "cpu":
            self.model.to(self.device)
        self.model.eval()
    
    @torch.inference_mode()
    def sequential_encode(self, video: torch.Tensor, **kwargs):
        # Should be tensor of shape (B, C, H, W)
        # Convert to (B, C//8 + 1, H//16, W//16)
        B, C, F, H, W = video.shape
        # check number of frames 
        # break up into micro batches
        out_mean: List[torch.Tensor] = []
        out_logvar: List[torch.Tensor] = []
        for i in tqdm(range(0, F, self.micro_batch_size), desc="Encoding video"):
            micro_batch = video[:, :, i:i+self.micro_batch_size, :, :]
            mean = self.model.encode(micro_batch).latent_dist.mean
            logvar = self.model.encode(micro_batch).latent_dist.logvar
            out_mean.append(mean)
            out_logvar.append(logvar)
        out_mean = torch.cat(out_mean, dim=0)
        out_logvar = torch.cat(out_logvar, dim=0)
        parameters = torch.cat([out_mean, out_logvar], dim=1)
        return parameters
    
    @torch.inference_mode()
    def encode(self, video: torch.Tensor, **kwargs):
        dist = self.model.encode(video).latent_dist
        return torch.cat([dist.mean, dist.logvar], dim=1)
    
    @torch.inference_mode()
    def decode(self, parameters: torch.Tensor, **kwargs):
        return self.model.decode(parameters.chunk(2, dim=1)[0])