"""Prétraitement : préparation, validation, découpage et conversion en tenseurs.

Dépend de src/data_generation.py (ACMG-19).
"""
import os

import pandas as pd
import torch

from src.data_generation import generate_dataset, save_csv

REQUIRED_COLUMNS = ["a", "b", "c", "regime"]


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie les colonnes, supprime les lignes vides, garde l'ordre temporel."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    return df.dropna().reset_index(drop=True)


def prepare_dataset(stationary: bool, directory: str = "data") -> pd.DataFrame:
    """Charge le CSV du dataset ; le génère d'abord s'il n'existe pas encore."""
    name = "stationary.csv" if stationary else "non_stationary.csv"
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        os.makedirs(directory, exist_ok=True)
        save_csv(generate_dataset(stationary=stationary), path)
    return clean_dataset(load_csv(path))


def split_stream(df: pd.DataFrame, train_ratio: float = 0.8):   
    cut = int(len(df) * train_ratio)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def to_tensors(df: pd.DataFrame):
    """X de forme (n, 2) = (a, b) ; y de forme (n, 1) = c."""
    X = torch.tensor(df[["a", "b"]].values, dtype=torch.float32)
    y = torch.tensor(df[["c"]].values, dtype=torch.float32)
    return X, y