import json
from pathlib import Path

import click
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
                    "-p",
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


def test_report(embeddings_datadir: Path, frequencies_datadir: Path, tmp_path: Path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as temp_dir:

        temp_path = Path(temp_dir)
        emb1_path = embeddings_datadir / "fasttext_ex.bin"
        freq_path = frequencies_datadir / "test_frequencies.json"

        # Test report generation for a single file that is not in a config file
        result = runner.invoke(cli.cli, ["report", emb1_path.resolve().as_posix()])
        expected_output_path = temp_path / "fasttext_ex_report.json"

        assert result.exit_code == 0
        assert expected_output_path.exists()

        with expected_output_path.open("r") as f:
            report = json.load(f)

        assert report == {
            "vector_size": 4,
            "n_elements": 2166,
            "default_frequency": 0,
            "mean_frequency": 0.0,
            "n_neighbors": 25,
            "mean_distance_neighbors": pytest.approx(0.0065, abs=1e-3),
            "mean_distance_first_neigbor": pytest.approx(0.0015, abs=1e-3),
        }

        # dd two embeddings and generate a comparison report
        for embedding_file, embedding_format in (
            ("fasttext_ex.bin", "bin"),
            ("word2vec_ex.bin", "word2vec"),
            ("embedding_test_1.json", "json"),
        ):
            result = runner.invoke(
                cli.cli,
                [
                    "add",
                    "-p",
                    (embeddings_datadir / embedding_file).resolve().as_posix(),
                    "-f",
                    embedding_format,
                    "--frequencies",
                    freq_path.resolve().as_posix(),
                    "-c",
                    "conf.yaml",
                ],
            )
            assert result.exit_code == 0

        result = runner.invoke(
            cli.cli, ["report", "fasttext_ex", "word2vec_ex", "-c", "conf.yaml"]
        )
        expected_output_path = temp_path / "fasttext_ex_word2vec_ex_report.json"

        assert result.exit_code == 0
        assert expected_output_path.exists()

        with expected_output_path.open("r") as f:
            report = json.load(f)

        assert report["embeddings"][0]["name"] == "fasttext_ex"
        assert report["embeddings"][1]["name"] == "word2vec_ex"
        assert report["neighborhoods_similarities_median"] == pytest.approx(0)

        # Test should fail if the embedding does not exists
        result = runner.invoke(cli.cli, ["report", "doesnotexists"])
        assert result.exit_code == 1
        assert "is not configured. Please add it" in result.stdout

        # Test should fail if the embedding does not exists
        result = runner.invoke(
            cli.cli,
            [
                "report",
                "fasttext_ex",
                "word2vec_ex",
                "embedding_test_1",
                "-c",
                "conf.yaml",
            ],
        )
        assert result.exit_code == 1
        assert "whereas only two can be compared" in result.stdout
