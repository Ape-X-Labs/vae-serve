from abc import ABC, abstractmethod
import torch


class BaseInference(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def _load_model(self):
        """Loads the model and any other necessary components into memory."""
        pass

    @torch.inference_mode()
    @abstractmethod
    def encode(self, **kwargs):
        pass
