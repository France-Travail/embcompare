import inspect
import sys
from pathlib import Path

import click
from streamlit.web import cli as stcli

from . import gui
from .config import CONFIG_EMBEDDINGS, load_config, save_config
from .load_utils import EMBEDDING_FORMATS, FREQUENCIES_FORMATS

GUI_DIR = Path(inspect.getfile(gui)).parent
APP_FILE = GUI_DIR / "app.py"

DEFAULT_CONFIG = "embcompare.yaml"
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)


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
    "--frequencies",
    type=click.Path(path_type=Path, resolve_path=True, exists=True, dir_okay=False),
)
@click.option(
    "--frequencies-format", type=click.Choice(FREQUENCIES_FORMATS, case_sensitive=False)
)
@click.option(
    "--labels",
    type=click.Path(path_type=Path, resolve_path=True, exists=True, dir_okay=False),
)
@click.option(
    "--labels-format", type=click.Choice(FREQUENCIES_FORMATS, case_sensitive=False)
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
    labels: Path = None,
    labels_format: Path = None,
    config_path: Path = None,
):
    config = load_config(config_path)

    if name is None:
        name = path.stem

    # Add format when not precised based on file extension
    if format is None:
        format = path.suffix[1:]

    if frequencies is not None:
        if frequencies_format is None:
            frequencies_format = frequencies.suffix[1:]

    if labels is not None:
        if labels_format is None:
            labels_format = labels.suffix[1:]

    # Get embedding entry in embedding configuration
    if CONFIG_EMBEDDINGS not in config:
        config[CONFIG_EMBEDDINGS] = {name: {}}

    if name not in config[CONFIG_EMBEDDINGS]:
        config[CONFIG_EMBEDDINGS][name] = {}

    embedding_conf = config[CONFIG_EMBEDDINGS][name]

    # Set embedding configuration
    embedding_conf.name = name
    embedding_conf.path = path.as_posix()

    if format:
        embedding_conf.format = format

    if frequencies:
        embedding_conf.frequencies = frequencies.as_posix()
    if frequencies_format:
        embedding_conf.frequencies_format = frequencies_format

    if labels:
        embedding_conf.labels = labels.as_posix()
    if labels_format:
        embedding_conf.labels_format = labels_format

    save_config(config, config_path)


@cli.command()
@click.argument("config", nargs=-1, type=click.Path(path_type=Path))
@click.option(
    "--log_level",
    type=click.Choice(["error", "warning", "info", "debug"], case_sensitive=False),
)
def gui(config: tuple, log_level: str = None):  # pragma: no cover
    config = config if config else (DEFAULT_CONFIG,)
    log_level = ["--logger.level", log_level] if log_level else []
    sys.argv = [
        "streamlit",
        "run",
        APP_FILE.as_posix(),
        *log_level,
        "--logger.messageFormat",
        LOG_FORMAT,
        "--",
        *config,
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":  # pragma: no cover
    cli()
