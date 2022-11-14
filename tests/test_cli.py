from pathlib import Path

import pytest
from click.testing import CliRunner
from embcompare import cli
from embcompare.config import CONFIG_EMBEDDINGS, load_config


@pytest.fixture
def embedding_test_1_path(embeddings_datadir: Path) -> Path:
    return embeddings_datadir / "embedding_test_1.json"


@pytest.fixture
def embedding_fasttext_path(embeddings_datadir: Path) -> Path:
    return embeddings_datadir / "fasttext_ex.bin"


def test_add(embedding_test_1_path: Path, tmp_path: Path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as temp_dir:

        temp_path = Path(temp_dir)
        config_path = temp_path / "embcompare.yaml"

        # Add embedding_test_1.json to config file
        result = runner.invoke(
            cli.cli, ["add", embedding_test_1_path.resolve().as_posix()]
        )

        assert result.exit_code == 0
        assert config_path.exists()

        config = load_config(config_path)
        assert (
            config[CONFIG_EMBEDDINGS][embedding_test_1_path.stem]["path"]
            == embedding_test_1_path.resolve().as_posix()
        )
