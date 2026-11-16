import pytest

from src.data_generation import generate_dataset
from src.preprocessing import clean_dataset, prepare_dataset, split_stream, to_tensors


def test_to_tensors_shapes():
    X, y = to_tensors(generate_dataset(True))
    assert X.shape == (3000, 2) and y.shape == (3000, 1)


def test_split_is_chronological():
    df = generate_dataset(False)
    train, test = split_stream(df, 0.8)
    assert len(train) == 2400 and len(test) == 600
    assert train["c"].iloc[-1] == df["c"].iloc[2399]


def test_missing_column_is_rejected():
    with pytest.raises(ValueError):
        clean_dataset(generate_dataset(True).drop(columns=["b"]))


def test_prepare_dataset_creates_csv(tmp_path):
    df = prepare_dataset(stationary=False, directory=str(tmp_path))
    assert len(df) == 3000
    assert (tmp_path / "non_stationary.csv").exists()
