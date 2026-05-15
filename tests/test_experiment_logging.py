import json

import yaml

from utils import ExperimentLogger


def test_experiment_logger_creates_run_artifacts(tmp_path):
    config = {"learning_rate": 0.01, "model": {"hidden_dim": 8}}

    logger = ExperimentLogger(
        config=config,
        seed=123,
        results_dir=tmp_path,
        run_name="smoke test",
    )

    assert logger.run_dir.parent == tmp_path
    assert logger.run_dir.name.endswith("-smoke-test")
    assert logger.metrics_path == logger.run_dir / "metrics.jsonl"

    with (logger.run_dir / "config.yaml").open(encoding="utf-8") as file:
        assert yaml.safe_load(file) == config

    with (logger.run_dir / "seed.json").open(encoding="utf-8") as file:
        assert json.load(file) == {"seed": 123}

    with (logger.run_dir / "environment.json").open(encoding="utf-8") as file:
        environment = json.load(file)

    assert "python" in environment
    assert "platform" in environment
    assert "git_commit" in environment
    assert "torch" in environment

    assert logger.metrics_path.read_text(encoding="utf-8") == ""


def test_log_metric_appends_json_lines(tmp_path):
    logger = ExperimentLogger(config={"batch_size": 4}, seed=5, results_dir=tmp_path)

    logger.log_metric(1, {"loss": 1.5, "accuracy": 0.25})
    logger.log_metric(2, {"loss": 1.25, "nested": {"accuracy": 0.5}})

    lines = logger.metrics_path.read_text(encoding="utf-8").splitlines()

    assert [json.loads(line) for line in lines] == [
        {"step": 1, "metrics": {"accuracy": 0.25, "loss": 1.5}},
        {"step": 2, "metrics": {"loss": 1.25, "nested": {"accuracy": 0.5}}},
    ]


def test_experiment_logger_creates_unique_run_directories(tmp_path, monkeypatch):
    class FixedDateTime:
        @staticmethod
        def now():
            return FixedDateTime()

        def strftime(self, _format):
            return "20260515-010203-000000"

    monkeypatch.setattr("utils.experiment_logging.datetime", FixedDateTime)

    first = ExperimentLogger(config={}, seed=1, results_dir=tmp_path, run_name="repeat")
    second = ExperimentLogger(config={}, seed=1, results_dir=tmp_path, run_name="repeat")

    assert first.run_dir != second.run_dir
    assert first.run_dir.name == "20260515-010203-000000-repeat"
    assert second.run_dir.name == "20260515-010203-000000-repeat-001"
