import json
import sys
from pathlib import Path

import pytest

from utils import generate_dataset


def test_save_stats_writes_json(tmp_path):
    stats = {
        "generated_at": "2025-01-01T00:00:00",
        "source": "synthetic",
        "samples_requested": 10,
    }
    out = tmp_path / "stats.json"
    generate_dataset.save_stats(stats, str(out))

    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["source"] == "synthetic"
    assert loaded["samples_requested"] == 10


class DummyYOLO:
    def __init__(self, output_dir, image_size, window_size, stride, val_split):
        self.output_dir = output_dir

    def generate_synthetic(self, n_samples, symbol):
        return {"train": int(n_samples * 0.8), "val": int(n_samples * 0.2)}

    def generate_from_csv(self, data_path, max_samples):
        return {"train": max_samples - 1, "val": 1}


class DummyLabeler:
    def __init__(self, forward_bars, threshold_pct):
        self.forward_bars = forward_bars


class DummyViT:
    def __init__(self, output_dir, image_size, window_size, stride, val_split, labeler, include_volume):
        self.output_dir = output_dir

    def generate_synthetic(self, n_samples, symbol, class_balance):
        return {"train": {"total": n_samples}, "val": {"total": 0}}

    def generate_from_csv(self, data_path, max_samples):
        return {"train": {"total": max_samples}, "val": {"total": 0}}


def test_generate_yolo_dataset_calls_synthetic(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_dataset, "YOLODatasetGenerator", DummyYOLO)

    stats = generate_dataset.generate_yolo_dataset(
        data_path=None,
        output_dir=str(tmp_path / "yolo"),
        samples=50,
        synthetic=True,
        image_size=128,
        window_size=20,
    )

    assert isinstance(stats, dict)
    assert stats["train"] == 40
    assert stats["val"] == 10


def test_generate_yolo_dataset_calls_from_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_dataset, "YOLODatasetGenerator", DummyYOLO)

    stats = generate_dataset.generate_yolo_dataset(
        data_path="some.csv",
        output_dir=str(tmp_path / "yolo"),
        samples=7,
        synthetic=False,
        image_size=128,
        window_size=20,
    )

    assert isinstance(stats, dict)
    assert stats["train"] == 6
    assert stats["val"] == 1


def test_generate_vit_dataset_calls_synthetic_and_from_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_dataset, "ViTDatasetGenerator", DummyViT)
    monkeypatch.setattr(generate_dataset, "FuturePriceLabeler", DummyLabeler)

    # synthetic
    stats_syn = generate_dataset.generate_vit_dataset(
        data_path=None,
        output_dir=str(tmp_path / "vit"),
        samples=30,
        synthetic=True,
        image_size=64,
        window_size=10,
    )
    assert isinstance(stats_syn, dict)
    assert stats_syn["train"]["total"] == 30

    # from csv
    stats_csv = generate_dataset.generate_vit_dataset(
        data_path="some.csv",
        output_dir=str(tmp_path / "vit"),
        samples=12,
        synthetic=False,
        image_size=64,
        window_size=10,
    )
    assert stats_csv["train"]["total"] == 12


def test_main_creates_generation_stats(monkeypatch, tmp_path):
    # Patch generators to fast dummies
    monkeypatch.setattr(generate_dataset, "YOLODatasetGenerator", DummyYOLO)
    monkeypatch.setattr(generate_dataset, "ViTDatasetGenerator", DummyViT)
    monkeypatch.setattr(generate_dataset, "FuturePriceLabeler", DummyLabeler)

    outdir = tmp_path / "out"
    args = ["generate_dataset.py", "--synthetic", "--samples", "5", "--output", str(outdir)]
    monkeypatch.setattr(sys, "argv", args)

    # Run main (should not raise)
    generate_dataset.main()

    stats_path = outdir / "generation_stats.json"
    assert stats_path.exists()
    data = json.loads(stats_path.read_text())
    assert data["source"] == "synthetic"
    assert "yolo" in data
    assert "vit" in data
