from src.data_generation import generate_dataset, transition_indices


def test_stationary_has_single_regime():
    df = generate_dataset(stationary=True)
    assert len(df) == 3000 and df["regime"].sum() == 0


def test_non_stationary_has_nine_transitions():
    df = generate_dataset(stationary=False)
    assert len(transition_indices()) == 9
    assert df["regime"].sum() == 1500          # 5 blocs de saturation sur 10


def test_reproducible():
    assert generate_dataset(False).equals(generate_dataset(False))

