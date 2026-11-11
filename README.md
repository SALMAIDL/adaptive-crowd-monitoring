# Adaptive Crowd Monitoring

Prédiction de l'affluence dans les espaces publics avec adaptation au concept drift.

## Objectifs
- Prédire l'affluence à court terme avec une erreur maîtrisée.
- Détecter la dérive des données (concept drift) en continu.
- Réentraîner et redéployer automatiquement le modèle au-delà d'un seuil de dérive.

## Contexte
Étude comparative de quatre architectures (MLP, RNN, LSTM, GRU) en apprentissage
online sur des données non stationnaires. Le GRU obtient une MSE finale de 0,09
contre 1,17 pour le MLP sur données stationnaires.

## Technologies envisagées
Python, PyTorch, FastAPI, Streamlit, pytest, Docker, GitHub Actions, MLflow,
Prometheus, Grafana, Kubernetes (minikube), Jira.

## Structure du projet
| Dossier | Rôle |
|---|---|
| `src/` | Code source de l'application et des modèles |
| `data/` | Données (non versionnées si volumineuses) |
| `models/` | Modèles entraînés |
| `notebooks/` | Notebooks d'expérimentation |
| `tests/` | Tests de l'application |
| `docs/` | Documentation |

## Équipe
- Assiya Kouraa
- Salma Idrissi Lamghari
- Salma Lafnager
- Soukaina

Encadrement : [Nom de l'enseignant] — ENSA Khouribga, module DevOps, 2026-2027.

## Gestion de projet
Backlog et suivi sur Jira (clé du projet : `ACMG`).# adaptive-crowd-monitoring
Prédiction de l'affluence dans les espaces publics avec MLP, RNN, LSTM et GRU, et adaptation au concept drift
