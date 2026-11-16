from src.alerts import count_alerts, saturation_alert


def test_alert_levels():
    assert saturation_alert(1.0) == "normal"
    assert saturation_alert(3.6) == "attention"
    assert saturation_alert(4.5) == "saturation"


def test_count_alerts():
    assert count_alerts([1.0, 4.3, 5.0, 2.0]) == 2