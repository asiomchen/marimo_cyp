"""Pull and pin the two source tables used by the CYP3A4 EDA.

Network access is explicit: use ``--refresh``. Without it, this script only
verifies the checked-in snapshots and their SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
import urllib.request
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import requests
from rdkit import Chem


Row = dict[str, Any]

OPENADMET_REVISION = "3ac9c5dbb83eec5780ec7fa511908698cfe1396d"
OPENADMET_FILENAME = "openadmet_train.csv"
OPENADMET_SHA256 = "b8f79addd266fb6f9f4c222c5e4e73d926362328b6a8d2841871a54e46bd2278"
OPENADMET_URL = (
    "https://huggingface.co/datasets/openadmet/cyp-challenge-train-test/resolve/"
    f"{OPENADMET_REVISION}/cyp-challenge-TRAIN_inhibition.csv?download=true"
)
CHEMBL_TARGET = "CHEMBL340"
CHEMBL_RELEASE = "ChEMBL_37"
CHEMBL_CLIENT = "chembl-webresource-client==0.10.9"
PAGE_SIZE = 100
WORKERS = 8

ACTIVITY_FIELDS = [
    "activity_id",
    "assay_chembl_id",
    "canonical_smiles",
    "document_chembl_id",
    "molecule_chembl_id",
    "standard_value",
]
ASSAY_FIELDS = [
    "assay_chembl_id",
    "confidence_score",
    "relationship_type",
    "target_chembl_id",
]
CHEMBL_FIELDS = [
    "smiles",
    "pIC50",
    "chembl_ids",
    "chembl_assays",
    "n_assays",
    "assay_pIC50_range",
]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonicalize(smiles: str) -> str | None:
    if not smiles or "." in smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def configure_client() -> Any:
    from chembl_webresource_client.settings import Settings

    settings = Settings.Instance()
    settings.CACHING = False
    settings.TIMEOUT = 120
    settings.NEW_CLIENT_TIMEOUT = 120
    settings.TOTAL_RETRIES = 5
    settings.BACKOFF_FACTOR = 2
    settings.MAX_LIMIT = PAGE_SIZE
    # Importing ``new_client`` retrieves the API schema, so keep this import
    # inside the explicit networked refresh path.
    from chembl_webresource_client.new_client import new_client

    return new_client


def activity_query(client: Any):
    return (
        client.activity.filter(
            target_chembl_id=CHEMBL_TARGET,
            standard_type="IC50",
            standard_units="nM",
            standard_relation="=",
            standard_flag=1,
            potential_duplicate=0,
            data_validity_comment__isnull=True,
            standard_value__isnull=False,
            canonical_smiles__isnull=False,
        )
        .only(ACTIVITY_FIELDS)
        .order_by("activity_id")
    )


def fetch_pages(query_factory: Callable[[], Any], total: int) -> list[Row]:
    bounds = [(start, min(start + PAGE_SIZE, total)) for start in range(0, total, PAGE_SIZE)]

    def fetch(bound: tuple[int, int]) -> list[Row]:
        start, stop = bound
        return [dict(row) for row in query_factory()[start:stop]]

    rows: list[Row] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for index, page in enumerate(executor.map(fetch, bounds), 1):
            rows.extend(page)
            if index % 10 == 0 or index == len(bounds):
                print(f"ChEMBL pages: {index}/{len(bounds)}", flush=True)
    return rows


def fetch_direct_assays(client: Any, assay_ids: list[str]) -> set[str]:
    batches = [assay_ids[start : start + PAGE_SIZE] for start in range(0, len(assay_ids), PAGE_SIZE)]

    def fetch(batch: list[str]) -> list[Row]:
        query = client.assay.filter(
            assay_chembl_id__in=batch,
            target_chembl_id=CHEMBL_TARGET,
            relationship_type="D",
            confidence_score=9,
        ).only(ASSAY_FIELDS)
        return [dict(row) for row in query]

    rows: list[Row] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for page in executor.map(fetch, batches):
            rows.extend(page)
    return {str(row["assay_chembl_id"]) for row in rows}


def aggregate_chembl(activities: list[Row], direct_assays: set[str]) -> list[Row]:
    by_structure_assay: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
    for row in activities:
        assay_id = str(row.get("assay_chembl_id") or "")
        if assay_id not in direct_assays:
            continue
        smiles = canonicalize(str(row.get("canonical_smiles") or ""))
        if smiles is None:
            continue
        try:
            value_nm = float(row["standard_value"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value_nm) or value_nm <= 0:
            continue
        by_structure_assay[(smiles, assay_id)].append(
            (value_nm, str(row["molecule_chembl_id"]))
        )

    by_structure: dict[str, list[Row]] = defaultdict(list)
    for (smiles, assay_id), values in sorted(by_structure_assay.items()):
        assay_ic50_nm = statistics.geometric_mean(value for value, _ in values)
        by_structure[smiles].append(
            {
                "assay": assay_id,
                "ic50_nm": assay_ic50_nm,
                "pIC50": 9.0 - math.log10(assay_ic50_nm),
                "ids": {identifier for _, identifier in values},
            }
        )

    compounds: list[Row] = []
    for smiles, assay_rows in sorted(by_structure.items()):
        assay_ic50_values = [float(row["ic50_nm"]) for row in assay_rows]
        assay_pic50_values = [float(row["pIC50"]) for row in assay_rows]
        compound_ic50_nm = statistics.geometric_mean(assay_ic50_values)
        compounds.append(
            {
                "smiles": smiles,
                "pIC50": 9.0 - math.log10(compound_ic50_nm),
                "chembl_ids": "|".join(sorted({identifier for row in assay_rows for identifier in row["ids"]})),
                "chembl_assays": "|".join(sorted(str(row["assay"]) for row in assay_rows)),
                "n_assays": len(assay_rows),
                "assay_pIC50_range": max(assay_pic50_values) - min(assay_pic50_values),
            }
        )
    return compounds


def write_csv(path: Path, rows: list[Row], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def refresh(root: Path) -> None:
    client = configure_client()
    destination = root / "data" / "source"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pull-sources-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        openadmet_path = staging / OPENADMET_FILENAME
        with urllib.request.urlopen(OPENADMET_URL, timeout=120) as response:
            openadmet_path.write_bytes(response.read())
        if digest(openadmet_path) != OPENADMET_SHA256:
            raise RuntimeError("OpenADMET SHA-256 does not match the pinned revision")

        status = requests.get("https://www.ebi.ac.uk/chembl/api/data/status.json", timeout=120)
        status.raise_for_status()
        release = status.json().get("chembl_db_version")
        if release != CHEMBL_RELEASE:
            raise RuntimeError(f"Expected {CHEMBL_RELEASE}, got {release}; review before refreshing")

        query_factory = lambda: activity_query(client)
        expected_activities = len(query_factory())
        activities = fetch_pages(query_factory, expected_activities)
        activities.sort(key=lambda row: int(row["activity_id"]))
        if (
            len(activities) != expected_activities
            or len({row["activity_id"] for row in activities}) != expected_activities
        ):
            raise RuntimeError("ChEMBL activity pagination returned incomplete or duplicate rows")
        assay_ids = sorted({str(row["assay_chembl_id"]) for row in activities})
        direct_assays = fetch_direct_assays(client, assay_ids)
        compounds = aggregate_chembl(activities, direct_assays)
        chembl_path = staging / "chembl_cyp3a4.csv"
        write_csv(chembl_path, compounds, CHEMBL_FIELDS)

        manifest = {
            "openadmet": {
                "revision": OPENADMET_REVISION,
                "url": OPENADMET_URL,
                "rows": sum(1 for _ in openadmet_path.open()) - 1,
                "sha256": digest(openadmet_path),
            },
            "chembl": {
                "release": CHEMBL_RELEASE,
                "target": CHEMBL_TARGET,
                "client": CHEMBL_CLIENT,
                "aggregation": {
                    "within_assay": "geometric_mean_IC50",
                    "across_assays": "geometric_mean_IC50",
                },
                "query": {
                    "standard_type": "IC50",
                    "standard_units": "nM",
                    "standard_relation": "=",
                    "standard_flag": 1,
                    "potential_duplicate": 0,
                    "data_validity_comment": None,
                    "assay_relationship_type": "D",
                    "assay_confidence_score": 9,
                },
                "candidate_activity_rows": len(activities),
                "direct_assays": len(direct_assays),
                "compounds": len(compounds),
                "sha256": digest(chembl_path),
            },
            "retrieved_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if destination.exists():
            backup = destination.with_name(destination.name + ".old")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(destination, backup)
            os.replace(staging, destination)
            shutil.rmtree(backup)
        else:
            os.replace(staging, destination)


def verify(root: Path) -> None:
    source = root / "data" / "source"
    manifest = json.loads((source / "manifest.json").read_text())
    expected = {
        source / OPENADMET_FILENAME: manifest["openadmet"]["sha256"],
        source / "chembl_cyp3a4.csv": manifest["chembl"]["sha256"],
    }
    for path, expected_hash in expected.items():
        if not path.exists() or digest(path) != expected_hash:
            raise RuntimeError(f"source verification failed: {path}")
    print(
        f"verified {manifest['openadmet']['rows']} OpenADMET rows and "
        f"{manifest['chembl']['compounds']} ChEMBL compounds"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="download and replace the pinned snapshots")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.refresh:
        refresh(root)
    verify(root)


if __name__ == "__main__":
    main()
