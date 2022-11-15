from pathlib import Path

import pytest
from click.testing import CliRunner
from embcompare import cli
from embcompare.config import CONFIG_EMBEDDINGS, load_config


def test_add(embeddings_datadir: Path, frequencies_datadir: Path, tmp_path: Path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as temp_dir:

        temp_path = Path(temp_dir)
        config_path = temp_path / "test.yaml"
        freq_path = frequencies_datadir / "test_frequencies.json"

        config_path.touch()

        for emb_file in (
            "embedding_test_1.json",
            "embedding_test_1.kv",
            "fasttext_ex.bin",
        ):
            emb_path = embeddings_datadir / emb_file
            # Add embedding_test_1.json to config file
            result = runner.invoke(
                cli.cli,
                [
                    "add",
                    emb_path.resolve().as_posix(),
                    "--frequencies",
                    freq_path.resolve().as_posix(),
                    "--labels",
                    freq_path.resolve().as_posix(),
                    "-c",
                    "test.yaml",
                ],
            )

            assert result.exit_code == 0
            assert config_path.exists()

            config = load_config(config_path)
            assert (
                config[CONFIG_EMBEDDINGS][emb_path.stem]["path"]
                == emb_path.resolve().as_posix()
            )
