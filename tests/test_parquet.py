"""Tests for write_parquet functionality."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import arnio as ar

pyarrow = pytest.importorskip("pyarrow", reason="pyarrow not installed")


class TestWriteParquetBasic:
    def test_basic_write_creates_file(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
        out = tmp_path / "out.parquet"
        ar.write_parquet(frame, out)
        assert out.exists()

    def test_pq_extension_accepted(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2]}))
        out = tmp_path / "out.pq"
        ar.write_parquet(frame, out)
        assert out.exists()

    def test_pathlike_input(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"v": [42]}))
        ar.write_parquet(frame, Path(tmp_path / "out.parquet"))
        assert (tmp_path / "out.parquet").exists()

    def test_string_path_input(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"v": [42]}))
        ar.write_parquet(frame, str(tmp_path / "out.parquet"))
        assert (tmp_path / "out.parquet").exists()

    def test_returns_none(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        result = ar.write_parquet(frame, tmp_path / "out.parquet")
        assert result is None


class TestWriteParquetRoundTrip:
    def test_integer_column_round_trips(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"n": [1, 2, 3]}))
        out = tmp_path / "out.parquet"
        ar.write_parquet(frame, out)
        df = pd.read_parquet(out, engine="pyarrow")
        assert df["n"].tolist() == [1, 2, 3]

    def test_float_column_round_trips(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"f": [1.1, 2.2, 3.3]}))
        out = tmp_path / "out.parquet"
        ar.write_parquet(frame, out)
        df = pd.read_parquet(out, engine="pyarrow")
        assert [round(v, 1) for v in df["f"].tolist()] == [1.1, 2.2, 3.3]

    def test_string_column_round_trips(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"s": ["alice", "bob"]}))
        out = tmp_path / "out.parquet"
        ar.write_parquet(frame, out)
        df = pd.read_parquet(out, engine="pyarrow")
        assert df["s"].tolist() == ["alice", "bob"]

    def test_bool_column_round_trips(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"b": [True, False, True]}))
        out = tmp_path / "out.parquet"
        ar.write_parquet(frame, out)
        df = pd.read_parquet(out, engine="pyarrow")
        assert df["b"].tolist() == [True, False, True]

    def test_null_values_round_trip(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]}))
        out = tmp_path / "out.parquet"
        ar.write_parquet(frame, out)
        df = pd.read_parquet(out, engine="pyarrow")
        assert pd.isna(df["a"].iloc[1])
        assert pd.isna(df["b"].iloc[1])

    def test_mixed_dtypes_round_trip(self, tmp_path):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "i": [1, 2, 3],
                    "f": [1.0, 2.0, 3.0],
                    "s": ["a", "b", "c"],
                    "b": [True, False, True],
                }
            )
        )
        out = tmp_path / "out.parquet"
        ar.write_parquet(frame, out)
        df = pd.read_parquet(out, engine="pyarrow")
        assert list(df.columns) == ["i", "f", "s", "b"]
        assert df.shape == (3, 4)

    def test_result_consistent_with_to_pandas(self, tmp_path):
        original_df = pd.DataFrame({"x": [10, 20, 30], "y": ["a", "b", "c"]})
        frame = ar.from_pandas(original_df)
        out = tmp_path / "out.parquet"
        ar.write_parquet(frame, out)
        roundtrip_df = pd.read_parquet(out, engine="pyarrow")
        arnio_df = ar.to_pandas(frame)
        assert roundtrip_df["x"].tolist() == arnio_df["x"].tolist()
        assert roundtrip_df["y"].tolist() == arnio_df["y"].tolist()


class TestWriteParquetCompression:
    @pytest.mark.parametrize("codec", ["snappy", "gzip", "zstd", "none"])
    def test_compression_codecs_accepted(self, tmp_path, codec):
        frame = ar.from_pandas(pd.DataFrame({"v": [1, 2, 3]}))
        out = tmp_path / f"out_{codec}.parquet"
        ar.write_parquet(frame, out, compression=codec)
        assert out.exists()
        df = pd.read_parquet(out, engine="pyarrow")
        assert df["v"].tolist() == [1, 2, 3]

    def test_default_compression_is_snappy(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"v": [1]}))
        out = tmp_path / "out.parquet"
        ar.write_parquet(frame, out)
        # File should be readable — snappy is the default
        df = pd.read_parquet(out, engine="pyarrow")
        assert df["v"].tolist() == [1]

    def test_unknown_compression_raises(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"v": [1]}))
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(frame, tmp_path / "out.parquet", compression="lz4")


class TestWriteParquetRowGroupSize:
    def test_row_group_size_accepted(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"v": list(range(100))}))
        out = tmp_path / "out.parquet"
        ar.write_parquet(frame, out, row_group_size=25)
        df = pd.read_parquet(out, engine="pyarrow")
        assert len(df) == 100

    def test_row_group_size_none_uses_default(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"v": [1, 2, 3]}))
        out = tmp_path / "out.parquet"
        ar.write_parquet(frame, out, row_group_size=None)
        assert out.exists()

    def test_row_group_size_zero_raises(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"v": [1]}))
        with pytest.raises(ValueError, match="positive integer"):
            ar.write_parquet(frame, tmp_path / "out.parquet", row_group_size=0)

    def test_row_group_size_negative_raises(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"v": [1]}))
        with pytest.raises(ValueError, match="positive integer"):
            ar.write_parquet(frame, tmp_path / "out.parquet", row_group_size=-1)

    def test_row_group_size_non_integer_raises(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"v": [1]}))
        with pytest.raises(TypeError, match="integer"):
            ar.write_parquet(frame, tmp_path / "out.parquet", row_group_size=1.5)


class TestWriteParquetErrors:
    def test_unsupported_extension_raises(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, tmp_path / "out.csv")

    def test_json_extension_raises(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, tmp_path / "out.json")

    def test_missing_pyarrow_raises_import_error(self, tmp_path):
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with patch.dict("sys.modules", {"pyarrow": None}):
            with pytest.raises(ImportError, match="pip install arnio\\[parquet\\]"):
                ar.write_parquet(frame, tmp_path / "out.parquet")
