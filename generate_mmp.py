"""Generate the MMP database and transform files consumed by ``notebook.py``."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from rdkit import Chem


Row = dict[str, Any]
PROPERTIES = ("Combined_pIC50", "OpenADMET_pIC50", "ChEMBL_pIC50")
ANCHORS = ("OCNT-2395548", "OCNT-2395632", "OCNT-2395554")
OPENADMET_FILENAME = "openadmet_train.csv"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_sources(root: Path) -> dict[str, Any]:
    source = root / "data" / "source"
    manifest = json.loads((source / "manifest.json").read_text())
    expected = {
        source / OPENADMET_FILENAME: manifest["openadmet"]["sha256"],
        source / "chembl_cyp3a4.csv": manifest["chembl"]["sha256"],
    }
    for path, expected_hash in expected.items():
        if not path.exists() or digest(path) != expected_hash:
            raise RuntimeError(f"source verification failed: {path}")
    return manifest


def canonicalize(smiles: str) -> str | None:
    if not smiles or "." in smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def read_openadmet(path: Path) -> list[Row]:
    endpoint = "CYP3A4_pIC50_direct_inhibition"
    low_name = f"{endpoint}_conf_low"
    high_name = f"{endpoint}_conf_high"
    std_name = f"{endpoint}_std"
    by_smiles: dict[str, list[Row]] = defaultdict(list)
    with path.open(newline="") as stream:
        for source in csv.DictReader(stream):
            try:
                value = float(source[endpoint])
                low = float(source[low_name])
                high = float(source[high_name])
                std = float(source[std_name])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(number) for number in (value, low, high, std)):
                continue
            if not low <= value <= high or std < 0:
                continue
            smiles = canonicalize(str(source.get("SMILES") or ""))
            if smiles is None:
                continue
            by_smiles[smiles].append(
                {
                    "id": source["Molecule_Name"],
                    "smiles": smiles,
                    "pIC50": value,
                    "conf_low": low,
                    "conf_high": high,
                    "std": std,
                    "ci_width": high - low,
                }
            )
    rows = [min(group, key=lambda row: (row["ci_width"], row["id"])) for group in by_smiles.values()]
    return sorted(rows, key=lambda row: str(row["id"]))


def read_chembl(path: Path) -> list[Row]:
    with path.open(newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def geometric_mean_pic50(values: Sequence[float]) -> float:
    """Return the pIC50 corresponding to the geometric mean IC50."""
    ic50_molar = [10.0 ** (-value) for value in values]
    return -math.log10(statistics.geometric_mean(ic50_molar))


def merge_sources(openadmet: list[Row], chembl: list[Row]) -> list[Row]:
    openadmet_by_smiles = {str(row["smiles"]): row for row in openadmet}
    chembl_by_smiles = {str(row["smiles"]): row for row in chembl}
    rows: list[Row] = []
    for smiles in sorted(openadmet_by_smiles.keys() | chembl_by_smiles.keys()):
        oa = openadmet_by_smiles.get(smiles)
        ch = chembl_by_smiles.get(smiles)
        identifier = str(oa["id"]) if oa else "CHEMBL_" + hashlib.sha256(smiles.encode()).hexdigest()[:20]
        oa_value: float | str = float(oa["pIC50"]) if oa else ""
        ch_value: float | str = float(ch["pIC50"]) if ch else ""
        source_values = [float(value) for value in (oa_value, ch_value) if value != ""]
        rows.append(
            {
                "id": identifier,
                "smiles": smiles,
                "Combined_pIC50": geometric_mean_pic50(source_values),
                "OpenADMET_pIC50": oa_value,
                "ChEMBL_pIC50": ch_value,
            }
        )
    return rows


def mmpdb_executable() -> str:
    adjacent = Path(sys.executable).with_name("mmpdb")
    executable = str(adjacent) if adjacent.exists() else shutil.which("mmpdb")
    if not executable:
        raise RuntimeError("mmpdb executable not found in the active environment")
    return executable


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def build_database(rows: Sequence[Row], output: Path) -> None:
    working = output.parent / ".mmp-work"
    working.mkdir(parents=True, exist_ok=True)
    smiles_path = working / "compounds.smi"
    properties_path = working / "properties.tsv"
    fragments_path = working / "compounds.fragdb"
    with smiles_path.open("w") as stream:
        for row in rows:
            stream.write(f"{row['smiles']}\t{row['id']}\n")
    with properties_path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["ID", *PROPERTIES])
        for row in rows:
            writer.writerow(
                [row["id"], *[row[name] if row[name] != "" else "*" for name in PROPERTIES]]
            )

    executable = mmpdb_executable()
    run(
        [
            executable,
            "fragment",
            "--num-cuts",
            "1",
            "--max-rotatable-bonds",
            "20",
            "--max-heavies",
            "100",
            "--salt-remover",
            "<none>",
            "--num-jobs",
            "1",
            "--output",
            str(fragments_path),
            str(smiles_path),
        ]
    )
    run(
        [
            executable,
            "index",
            str(fragments_path),
            "--properties",
            str(properties_path),
            "--max-variable-heavies",
            "10",
            "--max-variable-ratio",
            "0.4",
            "--min-radius",
            "0",
            "--max-radius",
            "3",
            "--title",
            "OpenADMET + ChEMBL CYP3A4 pIC50",
            "--output",
            str(output),
        ]
    )
    with sqlite3.connect(output) as connection:
        connection.execute("UPDATE dataset SET creation_date = '2026-09-29 00:00:00.000000'")
        connection.commit()
        connection.execute("VACUUM")
    shutil.rmtree(working)


def database_counts(database: Path) -> dict[str, int]:
    with sqlite3.connect(database) as connection:
        return {
            table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in ("compound", "rule", "pair", "rule_environment")
        }


def neighbor_count(database: Path, anchor_id: str) -> int:
    with sqlite3.connect(database) as connection:
        return int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT DISTINCT MIN(c1.public_id, c2.public_id), MAX(c1.public_id, c2.public_id)
                    FROM pair p
                    JOIN compound c1 ON c1.id = p.compound1_id
                    JOIN compound c2 ON c2.id = p.compound2_id
                    WHERE c1.public_id = ? OR c2.public_id = ?
                )
                """,
                (anchor_id, anchor_id),
            ).fetchone()[0]
        )


def generate(root: Path) -> None:
    source_manifest = verify_sources(root)
    source = root / "data" / "source"
    openadmet = read_openadmet(source / OPENADMET_FILENAME)
    chembl = read_chembl(source / "chembl_cyp3a4.csv")
    rows = merge_sources(openadmet, chembl)
    by_id = {str(row["id"]): row for row in rows}

    data_root = root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".generate-mmp-", dir=data_root) as temporary:
        staging = Path(temporary)
        database = staging / "cyp3a4.mmpdb"
        build_database(rows, database)

        executable = mmpdb_executable()
        queries: list[Row] = []
        for anchor_id in ANCHORS:
            anchor = by_id[anchor_id]
            for property_name in PROPERTIES:
                filename = f"transforms_{anchor_id}_{property_name}.tsv"
                transform_path = staging / filename
                run(
                    [
                        executable,
                        "transform",
                        str(database),
                        "--smiles",
                        str(anchor["smiles"]),
                        "--min-radius",
                        "1",
                        "--property",
                        property_name,
                        "--output",
                        str(transform_path),
                    ]
                )
                with transform_path.open() as stream:
                    products = max(sum(1 for _ in stream) - 1, 0)
                queries.append(
                    {
                        "id": anchor_id,
                        "smiles": anchor["smiles"],
                        "pIC50": anchor["OpenADMET_pIC50"],
                        "observed_neighbors": neighbor_count(database, anchor_id),
                        "file": filename,
                        "property": property_name,
                        "products": products,
                    }
                )

        (staging / "queries.json").write_text(json.dumps(queries, indent=2) + "\n")
        generated = {
            "aggregation": {
                "chembl_within_assay": "geometric_mean_IC50",
                "chembl_across_assays": "geometric_mean_IC50",
                "combined_sources": "geometric_mean_IC50",
            },
            "source_sha256": {
                "openadmet": source_manifest["openadmet"]["sha256"],
                "chembl": source_manifest["chembl"]["sha256"],
            },
            "curated_openadmet": len(openadmet),
            "curated_chembl": len(chembl),
            "combined_structures": len(rows),
            "database": database_counts(database),
            "transforms": {query["file"]: query["products"] for query in queries},
        }
        (staging / "manifest.json").write_text(json.dumps(generated, indent=2, sort_keys=True) + "\n")

        destination = data_root / "mmp"
        if destination.exists():
            backup = data_root / ".mmp-old"
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(destination, backup)
            os.replace(staging, destination)
            shutil.rmtree(backup)
        else:
            os.replace(staging, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    generate(args.root.resolve())


if __name__ == "__main__":
    main()
