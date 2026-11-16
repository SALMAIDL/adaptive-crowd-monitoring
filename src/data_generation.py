
"""Génération des datasets synthétiques de foule.

Régime d'attraction : c[t] = a[t] + b[t] + alpha * c[t-1]
Régime de saturation : c[t] = a[t] - b[t] - alpha * c[t-1]
"""
import numpy as np
import pandas as pd

from src.constants import (A_RANGE, ALPHA, B_RANGE, BLOCK_SIZE, N_SAMPLES,
                           NOISE_STD, SEED)


def generate_dataset(stationary: bool = True, n_samples: int = N_SAMPLES,
                     block_size: int = BLOCK_SIZE, alpha: float = ALPHA,
                     noise_std: float = NOISE_STD, seed: int = SEED) -> pd.DataFrame:
    """Dataset stationnaire (règle fixe) ou non stationnaire (blocs alternés).

    Colonne regime : 0 = attraction (règle add), 1 = saturation (règle sub).
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(*A_RANGE, n_samples)
    b = rng.uniform(*B_RANGE, n_samples)
    c = np.zeros(n_samples)
    regime = np.zeros(n_samples, dtype=int)
    previous = 0.0                                   # c[-1] = 0 : zone vide
    for t in range(n_samples):
        if not stationary and (t // block_size) % 2 == 1:
            regime[t] = 1                            # bloc impair : saturation
            value = a[t] - b[t] - alpha * previous
        else:
            value = a[t] + b[t] + alpha * previous
        value += rng.normal(0.0, noise_std)
        c[t] = previous = value
    return pd.DataFrame({"a": a, "b": b, "c": c, "regime": regime})


def transition_indices(n_samples: int = N_SAMPLES, block_size: int = BLOCK_SIZE):
    """Instants des changements de règle : 300, 600, ..., 2700 (9 transitions)."""
    return list(range(block_size, n_samples, block_size))


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
