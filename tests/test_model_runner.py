import pandas as pd

from app.model_runner import has_enough_history


def test_has_enough_history():
    df = pd.DataFrame({"RR": range(7)})

    assert has_enough_history(df, 7) is True
    assert has_enough_history(df, 8) is False
