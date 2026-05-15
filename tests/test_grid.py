import pytest

from training import coordinate_for_index, experiment_coordinates


def test_experiment_coordinates_cover_dataset_seed_beta_grid():
    config = _grid_config()

    coordinates = experiment_coordinates(config)

    assert len(coordinates) == 16
    assert coordinates[0].dataset_name == "mnist"
    assert coordinates[0].seed == 0
    assert coordinates[0].beta_inf == 0.01
    assert coordinates[0].beta_opt == 0.01
    assert coordinates[-1].dataset_name == "fashion_mnist"
    assert coordinates[-1].seed == 1
    assert coordinates[-1].beta_inf == 1.0
    assert coordinates[-1].beta_opt == 1.0


def test_coordinate_for_index_rejects_invalid_index():
    with pytest.raises(IndexError, match="outside"):
        coordinate_for_index(_grid_config(), 16)


def _grid_config():
    return {
        "datasets": [{"name": "mnist"}, {"name": "fashion_mnist"}],
        "seeds": [0, 1],
        "betas": {"values": [0.01, 1.0]},
    }
