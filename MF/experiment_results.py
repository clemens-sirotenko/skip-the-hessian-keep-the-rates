"""Lossless snapshots of the parallel NMF notebook's recorded results.

Snapshots preserve per-seed histories and experiment settings for replotting.
They do not contain final factor vectors or solver state for resuming an
interrupted solve. Only load pickle files produced by a trusted local run.
"""

from datetime import datetime
from pathlib import Path
import gzip
import pickle


RESULT_VARIABLES = (
    "experiment_config", "experiment_seeds", "methods_to_run", "plot_keys",
    "run_metadata", "effective_workers", "experiment_threads",
    "all_seeds_histories", "all_seeds_statuses", "all_seeds_timing",
    "all_seeds_cubic_diagnostics",
)


def save_experiment_snapshot(directory, namespace):
    """Save recorded arrays unchanged, including partial method collections.

    directory is a new export folder; namespace is the notebook's globals().
    Return the completed results.pkl.gz path. Existing snapshots are never replaced.
    Fast gzip compression keeps NumPy arrays lossless without expanding to text.
    A temporary file is renamed only after serialization finishes successfully.
    """
    snapshot = {
        "format": "parallel-nmf-results",
        "version": 1,
        "saved_at": datetime.now().astimezone().isoformat(),
        "state": {name: namespace[name] for name in RESULT_VARIABLES},
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "results.pkl.gz"
    if path.exists():
        raise FileExistsError(f"Snapshot already exists: {path}")
    temporary_path = directory / "results.pkl.gz.tmp"
    with gzip.open(temporary_path, "xb", compresslevel=1) as handle:
        pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.rename(path)
    return path


def load_experiment_snapshot(path):
    """Load a trusted results.pkl.gz (or older results.pkl), without running solvers.

    Restore with globals().update(...). Integer seed keys, NumPy arrays and
    their dtypes are retained, unlike JSON representations of plotted means.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        snapshot = pickle.load(handle)
    if snapshot.get("format") != "parallel-nmf-results" or snapshot.get("version") != 1:
        raise ValueError("Unsupported NMF snapshot format or version.")
    state = snapshot["state"]
    missing = set(RESULT_VARIABLES) - state.keys()
    if missing:
        raise ValueError(f"Incomplete NMF snapshot; missing {sorted(missing)}")
    return state
