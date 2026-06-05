import pickle
from pathlib import Path


def save_pickle(obj, path: str):  # type: ignore
    path: Path = Path(path)  # type: ignore
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
