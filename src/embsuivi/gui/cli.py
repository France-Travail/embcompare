import sys
from pathlib import Path
from typing import Union

import click
from omegaconf import OmegaConf
from streamlit.web import cli as stcli

from .load_utils import EMBEDDING_FORMATS, FREQUENCIES_FORMATS

GUI_DIR = Path(__file__).parent
APP_FILE = GUI_DIR / "app.py"

DEFAULT_CONFIG = "embsuivi.yaml"
CONFIG_EMBEDDINGS = "embeddings"


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "path",
    type=click.Path(path_type=Path, resolve_path=True, exists=True, dir_okay=False),
)
@click.option("-n", "--name")
@click.option(
    "-f", "--format", type=click.Choice(EMBEDDING_FORMATS, case_sensitive=False)
)
@click.option(
    "-f",
    "--frequencies",
    type=click.Path(path_type=Path, resolve_path=True, exists=True, dir_okay=False),
)
@click.option(
    "--frequencies-format", type=click.Choice(FREQUENCIES_FORMATS, case_sensitive=False)
)
@click.option(
    "-c",
    "--config",
    "config_path",
    default=DEFAULT_CONFIG,
    show_default=True,
    type=click.Path(path_type=Path),
)
def add(
    path: Path,
    name: str = None,
    format: str = None,
    frequencies: Path = None,
    frequencies_format: Path = None,
    config_path: Path = None,
):
    config = load_config(config_path)

    if name is None:
        name = path.stem

    if format is None:
        if path.suffix == ".bin":
            format = "fasttext"
        elif path.suffix == ".kv":
            format = "keyedvectors"
        elif path.suffix == ".json":
            format = "json"

    if frequencies is not None:
        if frequencies_format is None:
            if frequencies.suffix == ".json":
                frequencies_format = "json"

    if CONFIG_EMBEDDINGS not in config:
        config[CONFIG_EMBEDDINGS] = {"name": {}}

    elif name not in config[CONFIG_EMBEDDINGS]:
        config[CONFIG_EMBEDDINGS][name] = {}

    embedding_conf = config[CONFIG_EMBEDDINGS][name]

    embedding_conf.name = name
    embedding_conf.path = path.as_posix()

    if format:
        embedding_conf.format = format
    if frequencies:
        embedding_conf.frequencies = frequencies
    if frequencies_format:
        embedding_conf.frequencies_format = frequencies_format

    save_config(config, config_path)


@cli.command()
@click.argument("config", nargs=-1, type=click.Path(path_type=Path))
def gui(config: tuple):
    config = config if config else (DEFAULT_CONFIG,)
    sys.argv = ["streamlit", "run", APP_FILE.as_posix(), "--", *config]
    sys.exit(stcli.main())


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


if __name__ == "__main__":
    cli()
