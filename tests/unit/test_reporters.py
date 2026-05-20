"""Unit tests for src.data.eda.reporters."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eda.reporters import BaseReporter, CSVReporter, JSONReporter, MultiReporter


class TestReporterInheritance:
    """2.7: CSVReporter and JSONReporter must inherit from BaseReporter."""

    def test_csv_reporter_is_base_reporter(self) -> None:
        assert issubclass(CSVReporter, BaseReporter)

    def test_json_reporter_is_base_reporter(self) -> None:
        assert issubclass(JSONReporter, BaseReporter)


class TestCSVReporter:
    """Tests for CSVReporter output."""

    def test_generates_csv_file(self, tmp_path: Path) -> None:
        df = pd.DataFrame({
            "filename": ["a.jpg", "b.jpg"],
            "source": ["medetec", "fusc"],
            "mask_has_wound": [True, False],
            "mask_wound_percentage": [10.0, 0.0],
            "image_brightness_mean": [100.0, 120.0],
        })
        reporter = CSVReporter()
        result = reporter.generate(df, tmp_path)

        assert result["records"] == 2
        csv_path = Path(result["path"])
        assert csv_path.exists()
        assert csv_path.name == "metadata_enriched.csv"

    def test_csv_content_has_expected_columns(self, tmp_path: Path) -> None:
        df = pd.DataFrame({
            "filename": ["a.jpg"],
            "mask_has_wound": [True],
            "mask_wound_percentage": [5.0],
        })
        reporter = CSVReporter()
        reporter.generate(df, tmp_path)

        out_csv = tmp_path / "metadata_enriched.csv"
        read_df = pd.read_csv(out_csv)
        assert "filename" in read_df.columns
        assert "mask_has_wound" in read_df.columns


class TestJSONReporter:
    """Tests for JSONReporter output."""

    def test_generates_json_file(self, tmp_path: Path) -> None:
        df = pd.DataFrame({
            "source": ["medetec", "fusc"],
            "mask_has_wound": [True, False],
            "mask_wound_percentage": [10.0, 0.0],
            "image_brightness_mean": [100.0, 120.0],
        })
        reporter = JSONReporter()
        result = reporter.generate(df, tmp_path)

        json_path = Path(result["path"])
        assert json_path.exists()
        assert json_path.name == "statistics.json"
        assert "stats" in result
        assert result["stats"]["wound_general"]["count"] == 2

    def test_json_by_source(self, tmp_path: Path) -> None:
        df = pd.DataFrame({
            "source": ["medetec", "medetec"],
            "mask_has_wound": [True, True],
            "mask_wound_percentage": [10.0, 20.0],
            "image_brightness_mean": [100.0, 110.0],
        })
        reporter = JSONReporter()
        result = reporter.generate(df, tmp_path)

        by_source = result["stats"]["by_source"]
        assert "medetec" in by_source
        assert by_source["medetec"]["count"] == 2


class TestMultiReporter:
    """Tests for MultiReporter orchestration."""

    def test_multi_reporter_runs_all(self, tmp_path: Path) -> None:
        df = pd.DataFrame({
            "source": ["medetec"],
            "mask_has_wound": [True],
            "mask_wound_percentage": [10.0],
            "image_brightness_mean": [100.0],
        })
        multi = MultiReporter([CSVReporter(), JSONReporter()])
        results = multi.generate_all(df, tmp_path)

        assert len(results) == 2
        assert results[0]["records"] == 1
        assert "stats" in results[1]

    def test_multi_reporter_empty_list(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"source": ["medetec"], "mask_has_wound": [True], "mask_wound_percentage": [10.0], "image_brightness_mean": [100.0]})
        multi = MultiReporter([])
        results = multi.generate_all(df, tmp_path)
        assert results == []
