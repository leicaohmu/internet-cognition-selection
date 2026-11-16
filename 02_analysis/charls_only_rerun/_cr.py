import pandas as pd
import os

OLD = "/path/to/project/an2"


def merge(df, name):
    """Prepend non-CHARLS rows of the previous full-cohort file to a CHARLS-only frame."""
    p = os.path.join(OLD, name)
    if "cohort" not in df or not os.path.exists(p):
        return df
    try:
        o = pd.read_csv(p)
    except Exception:
        return df
    if "cohort" not in o:
        return df
    if set(df.cohort.dropna().unique()) - {"CHARLS"}:
        return df
    return pd.concat([o[o.cohort != "CHARLS"], df], ignore_index=True)


_orig = pd.DataFrame.to_csv


def to_csv(self, path=None, *a, **k):
    if isinstance(path, str) and not os.path.isabs(path) and path.endswith(".csv"):
        self = merge(self, path)
    return _orig(self, path, *a, **k)


pd.DataFrame.to_csv = to_csv
