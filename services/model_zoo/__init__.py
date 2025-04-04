import os
from ruamel.yaml import YAML
from services.model_zoo.register import (
    register_vae,
    register_text_encoder,
    VAE_MODELS,
    TEXT_ENCODER_MODELS,
)
from loguru import logger

# load the models from the models.yaml file
current_dir = os.path.dirname(os.path.abspath(__file__))
models_file = os.path.join(current_dir, "models.yaml")

yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
with open(models_file, "r") as f:
    models = yaml.load(f)

for model in models["vae"]:
    register_vae(
        model["repo_id"],
        model["name"],
        cache_dir=model.get("cache_dir"),
        directory=model.get("directory"),
    )

for model in models["text_encoder"]:
    register_text_encoder(
        model["repo_id"],
        model["name"],
        cache_dir=model.get("cache_dir"),
        directory=model.get("directory"),
    )

logger.info("Models registered")
for model in VAE_MODELS:
    logger.info(f"VAE: {model}")

for model in TEXT_ENCODER_MODELS:
    logger.info(f"TEXT_ENCODER: {model}")
