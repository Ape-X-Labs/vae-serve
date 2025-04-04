from services.inference.text_encoder import T5TextEncoder
from tests.conditional import run_if
import torch
from tests.runner import runner


def test_load_model_cpu():
    model = T5TextEncoder(model_name="wan2.1-t2v-14b", device="cpu")
    assert model is not None
    assert model.model is not None
    assert model.tokenizer is not None

@run_if(lambda: torch.cuda.is_available())
def test_load_model_cuda():
    model = T5TextEncoder(model_name="wan2.1-t2v-14b", device="cuda")
    assert model is not None
    assert model.model is not None
    assert model.tokenizer is not None

@run_if(lambda: torch.backends.mps.is_available())
def test_load_model_mps():
    model = T5TextEncoder(model_name="wan2.1-t2v-14b", device="mps")
    assert model is not None
    assert model.model is not None
    assert model.tokenizer is not None


@run_if(lambda: torch.backends.mps.is_available())
def test_encode_mps():
    model = T5TextEncoder(model_name="wan2.1-t2v-14b", device="mps")
    prompt = "A beautiful woman in a red dress"
    embeds = model.encode(prompt)
    assert embeds is not None
    assert embeds.shape[0] == 1
    assert embeds.shape[2] == 4096

def test_encode_cpu():
    model = T5TextEncoder(model_name="wan2.1-t2v-14b", device="cpu")
    prompt = "A beautiful woman in a red dress"
    embeds = model.encode(prompt)
    assert embeds is not None
    assert embeds.shape[0] == 1
    assert embeds.shape[2] == 4096

@run_if(lambda: torch.cuda.is_available())
def test_encode_cuda():
    model = T5TextEncoder(model_name="wan2.1-t2v-14b", device="cuda")
    prompt = "A beautiful woman in a red dress"
    embeds = model.encode(prompt)
    assert embeds is not None
    assert embeds.shape[0] == 1
    assert embeds.shape[2] == 4096
    # clear cuda cache
    torch.cuda.empty_cache()

@run_if(lambda: torch.backends.mps.is_available())
def test_encode_batch_mps():
    model = T5TextEncoder(model_name="wan2.1-t2v-14b", device="mps")
    prompts = ["A beautiful woman in a red dress", "A man in a blue shirt"]
    embeds = model.encode(prompts)
    assert embeds is not None
    assert embeds.shape[0] == 2
    assert embeds.shape[2] == 4096

@run_if(lambda: torch.cuda.is_available())
def test_encode_batch_cuda():
    model = T5TextEncoder(model_name="wan2.1-t2v-14b", device="cuda")
    prompts = ["A beautiful woman in a red dress", "A man in a blue shirt"]
    embeds = model.encode(prompts)
    assert embeds is not None
    assert embeds.shape[0] == 2
    assert embeds.shape[2] == 4096
    # clear cuda cache
    torch.cuda.empty_cache()
    

def test_encode_batch_cpu():
    model = T5TextEncoder(model_name="wan2.1-t2v-14b", device="cpu")
    prompts = ["A beautiful woman in a red dress", "A man in a blue shirt"]
    embeds = model.encode(prompts)
    assert embeds is not None
    assert embeds.shape[0] == 2
    assert embeds.shape[2] == 4096

if __name__ == "__main__":
    runner(
        [
            test_load_model_cpu,
            test_load_model_cuda,
            test_load_model_mps,
            test_encode_mps,
            test_encode_cuda,
            test_encode_batch_mps,
            test_encode_batch_cuda,
        ]
    )
