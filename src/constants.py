"""Constantes du projet (valeurs du rapport initial)."""
N_SAMPLES = 3000        # taille des datasets
BLOCK_SIZE = 300        # taille d'un bloc non stationnaire (9 transitions)
ALPHA = 0.3             # coefficient de persistance
NOISE_STD = 0.02        # bruit additif
SEED = 42               # graine aléatoire
A_RANGE = (-5.0, 5.0)   # signal d'afflux a[t]
B_RANGE = (0.0, 5.0)    # pression contextuelle b[t]

