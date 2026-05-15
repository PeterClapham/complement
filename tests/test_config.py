import pytest

from utils import load_config


def test_load_config_reads_yaml_mapping(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
training:
  epochs: 3
""".strip(),
        encoding="utf-8",
    )

    assert load_config(config_path) == {"seed": 42, "training": {"epochs": 3}}


def test_load_config_returns_empty_dict_for_empty_file(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    assert load_config(config_path) == {}


def test_load_config_rejects_non_mapping_yaml(tmp_path):
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(config_path)
