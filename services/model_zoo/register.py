import os
from huggingface_hub import snapshot_download
from ruamel.yaml import YAML
from loguru import logger
from tqdm.auto import tqdm
import argparse
import json
VAE_MODELS = {}
TEXT_ENCODER_MODELS = {}

current_dir = os.path.dirname(os.path.abspath(__file__))
models_file = os.path.join(current_dir, "models.yaml")

def register_vae(
    repo_id: str,
    model_name: str,
    cache_dir: str | None = None,
    directory: str | None = None,
    **kwargs,
):
    """The model must be available on Huggingface."""
    # check if the model is already registered and downloaded
    logger.info(f"Downloading VAE {model_name} from {repo_id}")
    allow_patterns = None
    if directory is not None:
        if not directory.endswith("/*"):
            allow_patterns = directory + "/*"

    folder_path = snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        tqdm_class=tqdm,
        allow_patterns=allow_patterns,
        **kwargs,
    )
    if directory is None:
        directory = ""
    config_path = os.path.join(folder_path, directory, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    config["repo_id"] = repo_id
    config["cache_dir"] = cache_dir
    config["directory"] = directory
    config['folder_path'] = os.path.join(folder_path, directory)
    global VAE_MODELS
    VAE_MODELS[model_name] = config
    
    logger.info(f"VAE {model_name} registered")

    # add to models.yaml
    yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
    with open(models_file, "r") as f:
        models = yaml.load(f)
    # check if the model is already registered
    model_names = [model["name"] for model in models["vae"]]
    if model_name in model_names:
        logger.info(f"VAE {model_name} already in models.yaml")
        return
    models["vae"].append(
        {
            "repo_id": repo_id,
            "model_name": model_name,
            "cache_dir": cache_dir,
            "dir": directory,
        }
    )
    with open(models_file, "w") as f:
        yaml.dump(models, f)
    

def register_text_encoder(
    repo_id: str,
    model_name: str,
    cache_dir: str | None = None,
    directory: str | None = None,
    **kwargs,
):
    """The model must be available on Huggingface."""
    logger.info(f"Downloading TEXT_ENCODER {model_name} from {repo_id}")
    

    allow_patterns = None
    if directory is not None:
        if not directory.endswith("/*"):
            allow_patterns = directory + "/*"
    
    folder_path = snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        tqdm_class=tqdm,
        allow_patterns=allow_patterns,
        **kwargs,
    )
    if directory is None:
        directory = ""
    config_path = os.path.join(folder_path, directory, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config["repo_id"] = repo_id
    config["cache_dir"] = cache_dir
    config["directory"] = directory
    config['folder_path'] = os.path.join(folder_path, directory)
    
    global TEXT_ENCODER_MODELS
    TEXT_ENCODER_MODELS[model_name] = config
    
    logger.info(f"TEXT_ENCODER {model_name} registered")
    # add to models.yaml
    yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
    with open(models_file, "r") as f:
        models = yaml.load(f)
    # check if the model is already registered
    
    model_names = [model["name"] for model in models["text_encoder"]]
    if model_name in model_names:
        logger.info(f"TEXT_ENCODER {model_name} already in models.yaml")
        return
    models["text_encoder"].append(
        {
            "repo_id": repo_id,
            "model_name": model_name,
            "cache_dir": cache_dir,
            "directory": directory,
        }
    )
    with open(models_file, "w") as f:
        yaml.dump(models, f)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="The model name to register"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="The repo id to register the model to",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="The cache directory to use"
    )
    parser.add_argument(
        "--directory", type=str, default=None, help="The directory to use"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vae",
        help="The model type to register",
        choices=["vae", "text_encoder"],
    )
    args = parser.parse_args()

    if args.model_type == "vae":
        register_vae(
            args.repo_id,
            args.model_name,
            cache_dir=args.cache_dir,
            directory=args.directory,
        )
    elif args.model_type == "text_encoder":
        register_text_encoder(
            args.repo_id,
            args.model_name,
            cache_dir=args.cache_dir,
            directory=args.directory,
        )
