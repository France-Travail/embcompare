from pathlib import Path

import click
import pytest
from click.exceptions import Abort
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

    # load_configs should merfe the configs
    expected_result = {"a": [{"b": 1}, {"c": 2}], "d": "zzz", "f": {"g": 4}}

    loaded_conf = config.load_configs(conf_path, conf2_path)
    assert expected_result == loaded_conf


def test_config_not_found(monkeypatch):
    def mockconfirm_aborted(*args, **kwargs):
        return False

    monkeypatch.setattr(click, "confirm", mockconfirm_aborted)

    with pytest.raises(FileNotFoundError):
        config.load_config("noway.yaml")
