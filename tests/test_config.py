from pathlib import Path

import click
import pytest
from embcompare import config


def test_config(tmp_path: Path):
    conf = {"a": [{"b": 1}, {"c": 2}], "d": "e", "f": 3}
    conf_path = tmp_path / "conf.yaml"

    # Save config
    config.save_config(conf, conf_path)
    assert conf_path.exists()

    # reload config
    loaded_conf = config.load_config(conf_path)
    assert conf == loaded_conf

    # Save another config
    conf2 = {"d": "zzz", "f": {"g": 4}}
    conf2_path = tmp_path / "conf2.yaml"
    config.save_config(conf2, conf2_path)

    # load_configs should merge the configs
    expected_result = {"a": [{"b": 1}, {"c": 2}], "d": "zzz", "f": {"g": 4}}

    loaded_conf = config.load_configs(conf_path, conf2_path)
    assert expected_result == loaded_conf


def test_config_auto_creation(monkeypatch, tmp_path: Path):
    # Monkeypath click.confirm to simulate a user abortion
    def mockconfirm_accepted(*args, **kwargs):
        return True

    monkeypatch.setattr(click, "confirm", mockconfirm_accepted)

    # One can also load a config file that does not exists, it will be
    # automaticaly crated
    loaded_conf = config.load_config(tmp_path / "auto.yaml", autocreate="confirm")
    assert loaded_conf == {config.CONFIG_EMBEDDINGS: {}}


def test_config_not_found(monkeypatch):
    # Monkeypath click.confirm to simulate a user abortion
    def mockconfirm_aborted(*args, **kwargs):
        return False

    monkeypatch.setattr(click, "confirm", mockconfirm_aborted)

    # Since the file does not exists it should return a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        config.load_config("noway.yaml", autocreate="confirm")


def test_add_to_config(embeddings_datadir: Path, frequencies_datadir: Path):
    conf = {}
    emb_path = embeddings_datadir / "embedding_test_1.json"
    freq_path = frequencies_datadir / "test_frequencies.json"
    labels_path = frequencies_datadir / "test_frequencies.json"

    config.add_to_config(
        conf,
        embedding_path=emb_path,
        frequencies_path=freq_path,
        labels_path=labels_path,
    )

    assert conf == {
        config.CONFIG_EMBEDDINGS: {
            "embedding_test_1": {
                "name": "embedding_test_1",
                "path": emb_path.resolve().as_posix(),
                "format": "json",
                "frequencies": freq_path.resolve().as_posix(),
                "frequencies_format": "json",
                "labels": labels_path.resolve().as_posix(),
                "labels_format": "json",
            }
        }
    }

    # Change only frequencies
    freq_path = frequencies_datadir / "test_frequencies_altered.pkl"

    config.add_to_config(
        conf,
        embedding_name="embedding_test_1",
        frequencies_path=freq_path,
    )

    assert conf == {
        config.CONFIG_EMBEDDINGS: {
            "embedding_test_1": {
                "name": "embedding_test_1",
                "path": emb_path.resolve().as_posix(),
                "format": "json",
                "frequencies": freq_path.resolve().as_posix(),
                "frequencies_format": "pkl",
                "labels": labels_path.resolve().as_posix(),
                "labels_format": "json",
            }
        }
    }
