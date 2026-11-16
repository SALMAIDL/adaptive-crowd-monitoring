"""Alertes de saturation affichées par le tableau de bord."""

SATURATION_THRESHOLD = 4.25   # seuil opérationnel de densité (voir le rapport initial)


def saturation_alert(prediction: float, threshold: float = SATURATION_THRESHOLD) -> str:
    """Retourne le niveau d'alerte : 'normal', 'attention' ou 'saturation'."""
    if prediction >= threshold:
        return "saturation"
    if prediction >= 0.8 * threshold:
        return "attention"
    return "normal"


def count_alerts(predictions, threshold: float = SATURATION_THRESHOLD) -> int:
    """Nombre de pas de temps en saturation."""
    return sum(1 for p in predictions if p >= threshold)