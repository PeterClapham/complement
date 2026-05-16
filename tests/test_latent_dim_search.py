import pytest

from training import (
    aggregate_latent_dimension_results,
    propose_next_dimensions,
    within_margin_of_error,
)


def test_aggregate_latent_dimension_results_reports_mean_and_error():
    rows = [
        {"latent_dim": 16, "final_validation_reconstruction": 4.0},
        {"latent_dim": 16, "final_validation_reconstruction": 6.0},
        {"latent_dim": 32, "final_validation_reconstruction": 3.0},
        {"latent_dim": 32, "final_validation_reconstruction": 3.0},
    ]

    aggregated = aggregate_latent_dimension_results(rows)

    assert aggregated[0]["latent_dim"] == 16
    assert aggregated[0]["mean_final_validation_reconstruction"] == 5.0
    assert aggregated[0]["standard_error"] > 0.0
    assert aggregated[1]["standard_error"] == 0.0


def test_propose_next_dimensions_refines_around_predicted_minimum():
    aggregated = [
        {"latent_dim": 16, "mean_final_validation_reconstruction": 5.0},
        {"latent_dim": 32, "mean_final_validation_reconstruction": 3.0},
        {"latent_dim": 48, "mean_final_validation_reconstruction": 2.0},
        {"latent_dim": 64, "mean_final_validation_reconstruction": 3.0},
        {"latent_dim": 80, "mean_final_validation_reconstruction": 4.0},
    ]

    dimensions, predicted_optimum = propose_next_dimensions(
        aggregated,
        current_dimensions=[16, 32, 48, 64, 80],
        array_size=5,
        low=16,
        high=80,
    )

    assert predicted_optimum == pytest.approx(48.0, abs=0.01)
    assert dimensions == [32, 40, 48, 56, 64]


def test_margin_of_error_detects_neighbor_overlap():
    aggregated = [
        {
            "latent_dim": 32,
            "mean_final_validation_reconstruction": 2.0,
            "standard_error": 0.2,
        },
        {
            "latent_dim": 48,
            "mean_final_validation_reconstruction": 1.9,
            "standard_error": 0.2,
        },
        {
            "latent_dim": 64,
            "mean_final_validation_reconstruction": 2.5,
            "standard_error": 0.1,
        },
    ]

    assert within_margin_of_error(aggregated)
