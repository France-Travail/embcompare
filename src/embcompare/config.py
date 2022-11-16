from pathlib import Path
from typing import Union

import click
from omegaconf import OmegaConf

CONFIG_EMBEDDINGS = "embeddings"


def load_config(config_path: Union[str, Path], autocreate: str = "no") -> OmegaConf:
    """Load a configuration file

    Args:
        config_path (Union[str, Path]): configuration file path
        autocreate (str, optional): Choose one of the following values :
            - "confirm" : ask the user if a config file should be automatically created.
            - "no" : do not autocreate a config file.
            - "yes" : do autocreate a config file.
            Defaults to "no".

    Raises:
        FileNotFoundError: Raises an error when the config file was not found.

    Returns:
        OmegaConf: An OmegaConf object
    """
    config_path = Path(config_path)

    if config_path.is_file():
        with config_path.open("r") as f:
            return OmegaConf.load(f)

    elif (
        autocreate == True
        or autocreate.lower() == "yes"
        or (
            isinstance(autocreate, str)
            and autocreate.lower() == "confirm"
            and click.confirm(
                f"{config_path} does not exists. Do you want to create it ?",
                default=True,
            )
        )
    ):
        config = OmegaConf.create({CONFIG_EMBEDDINGS: {}})
        save_config(config, config_path)

        return config
    else:
        raise FileNotFoundError(f"{config_path} does not exists.")


def save_config(config: Union[OmegaConf, dict], config_path: Union[str, Path]):
    """Save a configuration as a yaml file

    Args:
        config (Union[OmegaConf, dict]): configuration as a dict or OmegaConf object
        config_path (Union[str, Path]): configuration path
    """
    config_path = Path(config_path)

    with config_path.open("w") as f:
        OmegaConf.save(config=config, f=f)


def load_configs(*configs: Union[str, Path]) -> OmegaConf:
    """Load one or many yaml configuration files and merge them into a single
    OmegaConf object

    Returns:
        OmegaConf: A OmegaConf object
    """
    return OmegaConf.merge(*[load_config(path) for path in configs])


def add_to_config(
    config: Union[dict, OmegaConf],
    embedding_path: Path = None,
    embedding_name: str = None,
    embedding_format: str = None,
    frequencies_path: Path = None,
    frequencies_format: str = None,
    labels_path: Path = None,
    labels_format: str = None,
):
    """Add an embedding to a configuration file

    Args:
        config (Union[dict, OmegaConf]): Configuration
        embedding_path (Path): Embedding file path
        embedding_name (str, optional): Embedding name. Defaults to None.
        embedding_format (str, optional): Embedding file format. Defaults to None.
        frequencies (Path, optional): Frequencies file path. Defaults to None.
        frequencies_format (str, optional): Frequencies file format. Defaults to None.
        labels (Path, optional): Labels file path. Defaults to None.
        labels_format (Path, optional): Labels file format. Defaults to None.
        config_path (str, optional): Configuration file path. Defaults to None.
    """
    if embedding_name is None:
        embedding_name = embedding_path.stem

    # Add format when not precised based on file extension
    if embedding_format is None and embedding_path:
        embedding_format = embedding_path.suffix[1:]

    if frequencies_format is None and frequencies_path:
        frequencies_format = frequencies_path.suffix[1:]

    if labels_format is None and labels_path:
        labels_format = labels_path.suffix[1:]

    # Get embedding entry in embedding configuration
    if CONFIG_EMBEDDINGS not in config:
        config[CONFIG_EMBEDDINGS] = {}

    if embedding_name not in config[CONFIG_EMBEDDINGS]:
        config[CONFIG_EMBEDDINGS][embedding_name] = {}

    embedding_conf = config[CONFIG_EMBEDDINGS][embedding_name]

    # Set embedding configuration
    embedding_conf["name"] = embedding_name

    if embedding_path:
        embedding_conf["path"] = embedding_path.as_posix()

    if embedding_format:
        embedding_conf["format"] = embedding_format

    if frequencies_path:
        embedding_conf["frequencies"] = frequencies_path.as_posix()
    if frequencies_format:
        embedding_conf["frequencies_format"] = frequencies_format

    if labels_path:
        embedding_conf["labels"] = labels_path.as_posix()
    if labels_format:
        embedding_conf["labels_format"] = labels_format
