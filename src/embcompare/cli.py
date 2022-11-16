import os
import sys
from pathlib import Path
from typing import List

import click

from .config import CONFIG_EMBEDDINGS, add_to_config, load_config, save_config
from .embeddings_compare import EmbeddingComparison
from .gui import GUI_APP_FILE
from .load_utils import EMBEDDING_FORMATS, FREQUENCIES_FORMATS, load_embedding
from .reports import EmbeddingComparisonReport, EmbeddingReport

DEFAULT_CONFIG = "embcompare.yaml"
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)


@click.group()
def cli():
    """CLI main function used to group other commands as subcommands"""
    pass


@cli.command("add")
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
    help="Path to the configuration file",
)
def command_add(
    path: Path,
    name: str,
    format: str,
    frequencies: Path,
    frequencies_format: Path,
    labels: Path,
    labels_format: Path,
    config_path: Path,
):
    """Add an embedding to a configuration file"""
    config = load_config(config_path, autocreate="confirm")

    add_to_config(
        config,
        embedding_path=path,
        embedding_name=name,
        embedding_format=format,
        frequencies_path=frequencies,
        frequencies_format=frequencies_format,
        labels_path=labels,
        labels_format=labels_format,
    )

    save_config(config, config_path)


@cli.command()
@click.argument("embeddings", nargs=-1, required=True)
@click.option("-n", "--neighbors", default=25, type=click.IntRange(min=1))
@click.option("-o", "--output")
@click.option(
    "-c",
    "--config",
    "config_path",
    default=DEFAULT_CONFIG,
    show_default=True,
    type=click.Path(path_type=Path),
    help="Path to the configuration file",
)
def report(
    embeddings: List[str],
    neighbors: int,
    output: str,
    config_path: Path,
):
    try:
        config = load_config(config_path, autocreate="no")[CONFIG_EMBEDDINGS]
    except (FileNotFoundError, KeyError):
        config = {}

    embeddings_infos = {}

    for embedding in embeddings:
        # If the embedding is in the config, we get the embedding path and format
        # as well as frequencies path and format
        if embedding in config:
            embedding_path: str = config[embedding]["path"]
            embedding_format: str = config[embedding].get(
                "format", Path(embedding_path).suffix[1:]
            )
            frequencies_path: str = config[embedding].get("frequencies", None)
            frequencies_format: str = config[embedding].get("frequencies_format", None)

            if frequencies_format is None and frequencies_path:
                frequencies_format = Path(frequencies_path).suffix[1:]

            embeddings_infos[embedding] = {
                "embedding_path": embedding_path,
                "embedding_format": embedding_format,
                "frequencies_path": frequencies_path,
                "frequencies_format": frequencies_format,
            }

        # Otherwise if a path to an embedding has been given as input we load it
        elif os.path.isfile(embedding):
            embedding_path = Path(embedding)

            embeddings_infos[embedding_path.stem] = {
                "embedding_path": embedding,
                "embedding_format": embedding_path.suffix[1:],
                "frequencies_path": None,
                "frequencies_format": None,
            }

        else:
            raise click.ClickException(
                f"{embedding} is not configured. Please add it to a configuration file "
                f"thanks to the 'add' command."
            )

    if len(embeddings_infos) == 1:
        emb_infos = next(iter(embeddings_infos.values()))

        embedding = load_embedding(
            embedding_path=emb_infos["embedding_path"],
            embedding_format=emb_infos["embedding_format"],
            frequencies_path=emb_infos["frequencies_path"],
            frequencies_format=emb_infos["frequencies_format"],
        )

        if output is None:
            output = Path(emb_infos["embedding_path"]).stem + "_report.json"

        report = EmbeddingReport(embedding, n_neighbors=neighbors)
        report.to_json(output, indent=2)

    elif len(embeddings_infos) == 2:
        embeddings = {
            emb_name: load_embedding(
                embedding_path=emb_infos["embedding_path"],
                embedding_format=emb_infos["embedding_format"],
                frequencies_path=emb_infos["frequencies_path"],
                frequencies_format=emb_infos["frequencies_format"],
            )
            for emb_name, emb_infos in embeddings_infos.items()
        }

        if output is None:
            output = (
                "_".join(
                    [
                        Path(emb_infos["embedding_path"]).stem
                        for emb_infos in embeddings_infos.values()
                    ]
                )
                + "_report.json"
            )

        comparison = EmbeddingComparison(embeddings, n_neighbors=neighbors)
        report = EmbeddingComparisonReport(comparison)
        report.to_json(output, indent=2)

    else:
        raise click.ClickException(
            f"{len(embeddings_infos)} embeddings have been provided whereas only two can be "
            f"compared with a single call to the 'report' command"
        )


@cli.command(help="Start a streamlit app for embeddings comparison")
@click.argument("config", nargs=-1, type=click.Path(path_type=Path))
@click.option(
    "--log_level",
    type=click.Choice(["error", "warning", "info", "debug"], case_sensitive=False),
)
def gui(config: tuple, log_level: str):  # pragma: no cover
    """Start a streamlit app for embeddings comparison

    Args:
        config (tuple): configuration files paths
        log_level (str, optional): logging level. Defaults to None.

    Raises:
        click.ClickException: Raise an excpetion if gui dependencies are not installed.
    """
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        raise click.ClickException(
            "gui dependencies are not installed. "
            "Please install them by running : pip install embcompare[gui]"
        )

    config = config if config else (DEFAULT_CONFIG,)
    log_level = ["--logger.level", log_level] if log_level else []
    sys.argv = [
        "streamlit",
        "run",
        GUI_APP_FILE.as_posix(),
        *log_level,
        "--logger.messageFormat",
        LOG_FORMAT,
        "--",
        *config,
    ]
    sys.exit(stcli.main())
