"""Prepare saved data at the paths expected by the unchanged experiment notebooks.

Run from any directory with Python 3.12. This copies only the NN reference
snapshot; it performs no training and never replaces different existing data.
"""

import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root):
    root = Path(root).resolve()
    paper_dir = root / "NN" / "paper_run"
    manifest = json.loads((paper_dir / "manifest.json").read_text(encoding="utf-8"))
    source = paper_dir / "nn_results_new_v2.pkl"
    expected_hash = manifest["results"]["sha256"]
    if sha256(source) != expected_hash:
        raise ValueError("The NN paper-run file does not match its provenance manifest.")

    destination = root / "NN" / "corrected_results" / "paper_run" / source.name
    if not destination.resolve().is_relative_to(root):
        raise ValueError("The notebook data path resolves outside the repository.")
    if destination.exists():
        if sha256(destination) != expected_hash:
            raise FileExistsError(
                f"Different local results already exist at {destination}; left unchanged."
            )
        print("The saved NN paper run is already available at the notebook's expected path.")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    if sha256(destination) != expected_hash:
        raise OSError("The copied NN paper-run file failed its integrity check.")
    print("Prepared the saved NN paper run. No notebooks were modified.")


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parent)
