from pathlib import Path
from typing import Union

import click
from omegaconf import OmegaConf

CONFIG_EMBEDDINGS = "embeddings"


def load_config(config_path: Union[str, Path]) -> OmegaConf:
    config_path = Path(config_path)

    if config_path.is_file():
        with config_path.open("r") as f:
            return OmegaConf.load(f)

    elif click.confirm(
        f"{config_path} does not exists. Do you want to create it ?", default=True
    ):
        config = OmegaConf.create({CONFIG_EMBEDDINGS: {}})
        save_config(config, config_path)

        return config
    else:
        raise FileNotFoundError(f"{config_path} does not exists.")


def save_config(config, config_path: Union[str, Path]):
    config_path = Path(config_path)

    with config_path.open("w") as f:
        OmegaConf.save(config=config, f=f)


def load_configs(*configs: Union[str, Path]) -> OmegaConf:
    return OmegaConf.merge(*[load_config(path) for path in configs])
