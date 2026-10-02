# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "altair>=6.3.0",
#     "marimo",
#     "marimo-mmp==0.0.1",
#     "matplotlib==3.10.6",
#     "pandas==3.0.6",
#     "rdkit==2026.3.5",
#     "scipy>=1.17.1",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full", app_title="CYP3A4 matched-pair exploration")


@app.cell(hide_code=True)
def _():
    import base64
    import csv
    import json
    import math
    import sqlite3
    import statistics
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Draw, rdChemReactions, rdDepictor, rdFMCS
    from rdkit.Geometry import Point2D
    from scipy import stats as scipy_stats

    from marimo_mmp import TransformDataset, TransformGraph

    return (
        Chem,
        Draw,
        Path,
        Point2D,
        TransformDataset,
        TransformGraph,
        alt,
        base64,
        csv,
        json,
        math,
        mo,
        pd,
        rdChemReactions,
        rdDepictor,
        rdFMCS,
        sqlite3,
        statistics,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What can a chemist learn about CYP3A4 inhibition from matched molecular pairs?

    CYP3A4 metabolises a large share of marketed drugs, so inhibiting it is a
    classic source of drug–drug interactions. When a lead series inhibits CYP3A4,
    the medicinal chemist's question is practical: **which small edit will help?**

    But looking at a table and a couple of pairs is not fun, can we do it better? a graph maybe?

    A **matched molecular pair** (MMP) is two compounds that differ by one
    well-defined structural change, such as Cl → H at one position. Because everything else stays fixed, the change in activity can be attributed to that edit.
    This notebook extracts MMPs from the OpenADMET CYP3A4 dataset, supplemented by ChEMBL, and asks three questions:

    1. **Can a single edit change CYP3A4 inhibition a lot — beyond measurement noise?**
    2. **Do edits transfer?** Does the same swap do the same thing in different molecules, and if not, is that chemistry or noise?
    3. **What should I try on my own molecule?**

    **pIC50 = −log10(IC50 in mol/L)**: higher means stronger inhibition, and a
    change of 1 is a 10-fold change in IC50. ChEMBL values are geometric-mean IC50s within and across assays; for structures in both sources the combined value is the geometric mean of the two.
    """)
    return


@app.cell(hide_code=True)
def _(Path, csv, json, mo, sqlite3):
    study_root = Path(__file__).resolve().parent
    source_dir = study_root / "data" / "source"
    mmp_dir = study_root / "data" / "mmp"
    source_path = source_dir / "openadmet_train.csv"
    database_path = mmp_dir / "cyp3a4.mmpdb"
    query_path = mmp_dir / "queries.json"
    mo.stop(
        not all(_path.exists() for _path in (source_path, database_path, query_path)),
        mo.md(
            "Data is missing. Run `uv run --locked python pull_sources.py --refresh` and "
            "`uv run --locked python generate_mmp.py` from this directory."
        ),
    )

    _endpoint = "CYP3A4_pIC50_direct_inhibition"
    _low = f"{_endpoint}_conf_low"
    _high = f"{_endpoint}_conf_high"
    _std = f"{_endpoint}_std"
    measurements = []
    with source_path.open(newline="") as _stream:
        for _row in csv.DictReader(_stream):
            try:
                _value = float(_row[_endpoint])
                _lo = float(_row[_low])
                _hi = float(_row[_high])
                _sigma = float(_row[_std])
            except (KeyError, TypeError, ValueError):
                continue
            measurements.append(
                {
                    "id": _row["Molecule_Name"],
                    "smiles": _row["SMILES"],
                    "pIC50": _value,
                    "conf_low": _lo,
                    "conf_high": _hi,
                    "std": _sigma,
                    "ci_width": _hi - _lo,
                }
            )
    measurement_by_id = {_row["id"]: _row for _row in measurements}
    queries = json.loads(query_path.read_text())
    generated_manifest = json.loads((mmp_dir / "manifest.json").read_text())
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as _connection:
        mmp_connected_structures = _connection.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT compound1_id AS compound_id FROM pair
                UNION
                SELECT compound2_id AS compound_id FROM pair
            )
            """
        ).fetchone()[0]
    return (
        database_path,
        generated_manifest,
        measurement_by_id,
        measurements,
        mmp_connected_structures,
        mmp_dir,
        queries,
    )


@app.cell(hide_code=True)
def _(generated_manifest, mmp_connected_structures, mo):
    _openadmet_structures = generated_manifest["curated_openadmet"]
    _chembl_structures = generated_manifest["curated_chembl"]
    _combined_structures = generated_manifest["combined_structures"]
    _shared_structures = (
        _openadmet_structures + _chembl_structures - _combined_structures
    )
    _unpaired_structures = _combined_structures - mmp_connected_structures

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        value=f"{_openadmet_structures:,}",
                        label="OpenADMET structures",
                        caption="Usable CYP3A4 pIC50 values with reported activity intervals",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{_chembl_structures:,}",
                        label="ChEMBL structures",
                        caption="Curated compounds after geometric-mean assay aggregation",
                        bordered=True,
                    ),
                ],
                widths="equal",
            ),
            mo.hstack(
                [
                    mo.stat(
                        value=f"{_combined_structures:,}",
                        label="Unique structures after merging",
                        caption="OpenADMET ∪ ChEMBL after canonical-structure deduplication",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{mmp_connected_structures:,}",
                        label="Structures connected by MMPs",
                        caption="Compounds participating in at least one matched molecular pair",
                        bordered=True,
                    ),
                ],
                widths="equal",
            ),
            mo.md(
                f"The two sources share **{_shared_structures:,} structures**, so the "
                f"merged total is smaller than their sum. Of the "
                f"**{_combined_structures:,} unique structures**, "
                f"**{mmp_connected_structures:,}** form at least one matched pair; "
                f"the remaining **{_unpaired_structures:,}** have no partner under "
                f"the current fragmentation settings."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(database_path, measurement_by_id, sqlite3):
    _sql = """
        SELECT DISTINCT
            r.id,
            fs.smiles,
            ts.smiles,
            c1.public_id,
            c2.public_id
        FROM pair p
        JOIN rule_environment re ON re.id = p.rule_environment_id
        JOIN rule r ON r.id = re.rule_id
        JOIN rule_smiles fs ON fs.id = r.from_smiles_id
        JOIN rule_smiles ts ON ts.id = r.to_smiles_id
        JOIN compound c1 ON c1.id = p.compound1_id
        JOIN compound c2 ON c2.id = p.compound2_id
        WHERE re.radius = 0
          AND c1.public_id LIKE 'OCNT-%'
          AND c2.public_id LIKE 'OCNT-%'
        ORDER BY r.id, c1.public_id, c2.public_id
    """
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as _connection:
        _database_pairs = _connection.execute(_sql).fetchall()

    pair_rows = []
    _seen_rule_pairs = set()
    for _rule_id, _from_fragment, _to_fragment, _from_id, _to_id in _database_pairs:
        _key = (_rule_id, _from_id, _to_id)
        if _key in _seen_rule_pairs:
            continue
        _seen_rule_pairs.add(_key)
        _a = measurement_by_id[_from_id]
        _b = measurement_by_id[_to_id]
        _bound_low = _b["conf_low"] - _a["conf_high"]
        _bound_high = _b["conf_high"] - _a["conf_low"]
        pair_rows.append(
            {
                "rule_id": _rule_id,
                "from_id": _from_id,
                "to_id": _to_id,
                "from_fragment": _from_fragment,
                "to_fragment": _to_fragment,
                "from_pIC50": _a["pIC50"],
                "to_pIC50": _b["pIC50"],
                "delta_pIC50": _b["pIC50"] - _a["pIC50"],
                "delta_bound_low": _bound_low,
                "delta_bound_high": _bound_high,
                "interval_separated": _bound_low > 0 or _bound_high < 0,
                "max_ci_width": max(_a["ci_width"], _b["ci_width"]),
            }
        )
    unique_molecular_pairs = len(
        {tuple(sorted((_row["from_id"], _row["to_id"]))) for _row in pair_rows}
    )
    return pair_rows, unique_molecular_pairs


@app.cell(hide_code=True)
def _(mo, unique_molecular_pairs):
    mo.md(f"""
    ## 1 · Can a single edit change CYP3A4 inhibition a lot?

    The database contains **{unique_molecular_pairs} distinct OpenADMET molecular
    pairs** under the chosen fragmentation settings. A cliff is an observed pair
    whose absolute activity change exceeds the selected threshold.

    The interval bounds shown below combine the reported endpoints:
    `[B_low − A_high, B_high − A_low]`.

    /// details | How the matched-pair database was generated

    `generate_mmp.py` builds `data/mmp/cyp3a4.mmpdb` with mmpdb from the merged
    OpenADMET + ChEMBL structures (canonical RDKit SMILES, single-component only):

    - `mmpdb fragment --num-cuts 1 --max-heavies 100 --max-rotatable-bonds 20
    - `mmpdb index --max-variable-heavies 10 --max-variable-ratio 0.4 --min-radius 0 --max-radius 3`
    - Properties: `Combined_pIC50`, `OpenADMET_pIC50`, `ChEMBL_pIC50`.
    - `mmpdb transform --min-radius 1` for each anchor and property feeds section 3.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cliff_threshold = mo.ui.slider(
        0.3, 2.5, value=1.0, step=0.1, debounce=True, label="Minimum |ΔpIC50|"
    )
    width_threshold = mo.ui.slider(
        0.2, 4.0, value=2.0, step=0.1, debounce=True, label="Maximum interval width"
    )
    separated_only = mo.ui.checkbox(
        value=False, label="Require non-overlapping A/B activity intervals"
    )
    mo.hstack([cliff_threshold, width_threshold, separated_only], widths="equal")
    return cliff_threshold, separated_only, width_threshold


@app.cell(hide_code=True)
def _(cliff_threshold, pair_rows, separated_only, width_threshold):
    eligible_pairs = [
        _row for _row in pair_rows if _row["max_ci_width"] <= width_threshold.value
    ]
    cliffs = [
        _row
        for _row in eligible_pairs
        if abs(_row["delta_pIC50"]) >= cliff_threshold.value
        and (not separated_only.value or _row["interval_separated"])
    ]
    cliffs.sort(key=lambda _row: -abs(_row["delta_pIC50"]))


    def _unique_pairs(_rows):
        """Collapse rule-level rows to one row per directed (from_id, to_id) pair."""
        return list({(_row["from_id"], _row["to_id"]): _row for _row in _rows}.values())


    unique_eligible_pairs = _unique_pairs(eligible_pairs)
    unique_cliffs = _unique_pairs(cliffs)
    return cliffs, unique_cliffs, unique_eligible_pairs


@app.cell(hide_code=True)
def _(cliffs, mo, pair_rows, unique_cliffs, unique_eligible_pairs):
    mo.callout(
        mo.md(
            f"**{len(unique_cliffs)} candidate cliffs** among "
            f"**{len(unique_eligible_pairs)} matched pairs** "
            f"({len({(_row['from_id'], _row['to_id']) for _row in pair_rows})} before filtering). "
            f"The table below lists {len(cliffs)} rule-level rows, because one pair can "
            f"match several fragmentation rules. Try changing the ranges to see how it "
            f"will affect number of outliers (cliffs)"
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(
    alt,
    cliff_threshold,
    measurements,
    mo,
    pd,
    unique_cliffs,
    unique_eligible_pairs,
    width_threshold,
):
    _bin_counts = pd.Series(
        [_row["ci_width"] for _row in measurements]
    ).value_counts(bins=30, sort=False)
    _interval_bins = pd.DataFrame(
        {
            "bin_start": [_interval.left for _interval in _bin_counts.index],
            "bin_end": [_interval.right for _interval in _bin_counts.index],
            "molecules": _bin_counts.to_numpy(),
        }
    )
    _bars = (
        alt.Chart(_interval_bins)
        .mark_bar(
            color="#287d8e",
            stroke="white",
            strokeWidth=1,
            cornerRadiusTopLeft=2,
            cornerRadiusTopRight=2,
        )
        .encode(
            x=alt.X("bin_start:Q", title="Reported interval width (pIC50)"),
            x2="bin_end:Q",
            y=alt.Y("molecules:Q", title="Molecules", scale=alt.Scale(domainMin=0)),
            y2=alt.datum(0),
            tooltip=[
                alt.Tooltip("bin_start:Q", title="Width from", format=".2f"),
                alt.Tooltip("bin_end:Q", title="Width to", format=".2f"),
                alt.Tooltip("molecules:Q", title="Molecules"),
            ],
        )
    )
    _cutoff = alt.Chart(
        pd.DataFrame(
            {
                "x": [width_threshold.value],
                "label": [f"maximum = {width_threshold.value:.1f}"],
            }
        )
    )
    _cutoff_rule = _cutoff.mark_rule(
        color="#bb4b31", strokeDash=[4, 3], strokeWidth=1.5
    ).encode(x="x:Q")
    _cutoff_label = _cutoff.mark_text(
        align="left", dx=4, color="#555555", fontSize=11
    ).encode(x="x:Q", y=alt.value(10), text="label:N")
    _histogram = mo.ui.altair_chart(
        (_bars + _cutoff_rule + _cutoff_label).properties(
            title="Measurement precision varies widely", width=420, height=320
        ),
        chart_selection=False,
        legend_selection=False,
    )

    _cliff_keys = {(_row["from_id"], _row["to_id"]) for _row in unique_cliffs}
    cliff_points = pd.DataFrame(
        [
            {
                "from_id": _row["from_id"],
                "to_id": _row["to_id"],
                "from_pIC50": _row["from_pIC50"],
                "to_pIC50": _row["to_pIC50"],
                "delta_pIC50": _row["delta_pIC50"],
                "max_ci_width": _row["max_ci_width"],
                "status": (
                    "candidate cliff"
                    if (_row["from_id"], _row["to_id"]) in _cliff_keys
                    else "below threshold"
                ),
            }
            for _row in unique_eligible_pairs
        ]
    ).sort_values("status", ascending=False)
    _limits = [1.0, 7.6]
    _guides = pd.DataFrame(
        [
            {"x": _x, "y": _x + _offset, "guide": _guide}
            for _guide, _offset in (
                ("parity", 0.0),
                ("+threshold", cliff_threshold.value),
                ("−threshold", -cliff_threshold.value),
            )
            for _x in _limits
        ]
    )
    _scale = alt.Scale(domain=_limits, nice=False)
    _is_parity = alt.datum.guide == "parity"
    _guide_lines = (
        alt.Chart(_guides)
        .mark_line(strokeWidth=1, clip=True)
        .encode(
            x=alt.X("x:Q", scale=_scale),
            y=alt.Y("y:Q", scale=_scale),
            detail="guide:N",
            color=alt.condition(_is_parity, alt.value("#555555"), alt.value("#bb4b31")),
            strokeDash=alt.condition(_is_parity, alt.value([1, 0]), alt.value([4, 3])),
        )
    )
    _is_cliff = alt.datum.status == "candidate cliff"
    _dots = (
        alt.Chart(cliff_points)
        .mark_circle(stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("from_pIC50:Q", title="From pIC50", scale=_scale),
            y=alt.Y("to_pIC50:Q", title="To pIC50", scale=_scale),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(
                    domain=["below threshold", "candidate cliff"],
                    range=["#a7b1b7", "#bb4b31"],
                ),
                legend=alt.Legend(title=None, orient="top-left"),
            ),
            size=alt.condition(_is_cliff, alt.value(80), alt.value(50)),
            opacity=alt.condition(_is_cliff, alt.value(0.85), alt.value(0.45)),
            tooltip=[
                alt.Tooltip("from_id:N", title="A"),
                alt.Tooltip("to_id:N", title="B"),
                alt.Tooltip("from_pIC50:Q", title="A pIC50", format=".2f"),
                alt.Tooltip("to_pIC50:Q", title="B pIC50", format=".2f"),
                alt.Tooltip("delta_pIC50:Q", title="ΔpIC50 (B − A)", format="+.2f"),
                alt.Tooltip("max_ci_width:Q", title="Max interval width", format=".2f"),
            ],
        )
        .add_params(alt.selection_interval(name="pair_brush"))
    )
    cliff_scatter = mo.ui.altair_chart(
        (_guide_lines + _dots).properties(
            title="Matched-pair activity changes", width=420, height=320
        )
    )
    mo.hstack([_histogram, cliff_scatter], widths="equal", align="center")
    return


@app.cell(hide_code=True)
def _(cliffs, mo):
    cliff_table = mo.ui.table(
        cliffs,
        selection="single",
        page_size=7,
        show_download=True,
        label="Select a measured pair",
    )
    cliff_table
    return (cliff_table,)


@app.cell(hide_code=True)
def _(cliff_table, cliffs, mo, render_matched_pair):
    chosen_pair = (
        cliff_table.value[0]
        if cliff_table.value
        else (cliffs[0] if cliffs else None)
    )
    pair_display = (
        mo.callout("No pair meets these settings.", kind="warn")
        if chosen_pair is None
        else render_matched_pair(chosen_pair)
    )
    pair_display
    return


@app.cell(hide_code=True)
def _(cliff_threshold, mo, pair_rows, unique_cliffs, unique_eligible_pairs):
    _robust = list(
        {
            (_row["from_id"], _row["to_id"]): _row
            for _row in pair_rows
            if abs(_row["delta_pIC50"]) >= 1.0
            and _row["interval_separated"]
            and _row["max_ci_width"] <= 1.0
        }.values()
    )
    robust_cliffs = sorted(_robust, key=lambda _row: -abs(_row["delta_pIC50"]))
    _all_pairs = {(_row["from_id"], _row["to_id"]) for _row in pair_rows}
    _largest = robust_cliffs[0] if robust_cliffs else None
    mo.callout(
        mo.md(
            "Single edits can change CYP3A4 inhibition by more "
            "than 10-fold. "
            f"With the current settings, **{len(unique_cliffs)} of {len(unique_eligible_pairs)}** matched pairs change by ≥ "
            f"{cliff_threshold.value:.1f} log units. "
            f"Under more strict criteria — **|ΔpIC50| ≥ 1**, non-overlapping activity intervals, and both intervals narrower than 1 log unit — **{len(robust_cliffs)} of {len(_all_pairs)}** "
            f"pairs ({len(robust_cliffs) / len(_all_pairs):.0%}) remain, so they cannot "
            "be explained by measurement uncertainty. "
            + (
                f"The largest is **{_largest['from_id']} → {_largest['to_id']}** "
                f"({_largest['delta_pIC50']:+.2f})."
                if _largest
                else ""
            )
        ),
        kind="success",
    )
    return (robust_cliffs,)


@app.cell(hide_code=True)
def _(mo, robust_cliffs):
    robust_cliff_table = mo.ui.table(
        robust_cliffs,
        selection="single",
        page_size=7,
        show_download=True,
        label="Select a pair that passes the stricter criteria",
    )
    robust_cliff_table
    return (robust_cliff_table,)


@app.cell(hide_code=True)
def _(mo, render_matched_pair, robust_cliff_table, robust_cliffs):
    chosen_robust_pair = (
        robust_cliff_table.value[0]
        if robust_cliff_table.value
        else (robust_cliffs[0] if robust_cliffs else None)
    )
    (
        mo.callout("No pair passes the stricter criteria.", kind="warn")
        if chosen_robust_pair is None
        else render_matched_pair(chosen_robust_pair)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · Do edits transfer between molecules?

    A single matched pair can be noise. If the same fragment swap shows the same
    effect across many unrelated pairs, it is more likely to be a real
    structure–activity trend you can reuse in design. Statistics follow
    Benjamini & Hochberg (1995) for multiple testing.

    Each row below is one **transformation** (from-fragment → to-fragment), optionally
    restricted to a chemical **environment**: radius 0 pools every occurrence of the
    swap, while radius 1–3 also requires the atoms around the attachment point to match.
    A higher radius gives more specific chemistry but fewer pairs.

    This section uses **Combined pIC50** (OpenADMET and ChEMBL), because OpenADMET
    alone has only a handful of repeated swaps. ChEMBL values come from mixed assays,
    have no reported intervals, and often sit at assay cut-offs (e.g. 20 µM), so many
    pairs show exactly zero change.

    How to read the table:

    - **ΔpIC50** is *to − from*; positive means the swap increases inhibition.
    - **up / down / flat** count pairs with ΔpIC50 > +0.1, < −0.1, or in between.
    - **agreement** is the share of non-flat pairs that move in the majority direction.
    - **sign_p** is an exact two-sided sign test on up vs down; **q** applies a
      Benjamini–Hochberg correction across all rows, since hundreds of rules are tested.
    - **trend** is *increases* / *decreases* when q < 0.05, *leans up* / *leans down*
      when only the uncorrected sign_p < 0.05, and otherwise *no consistent direction*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    rule_radius = mo.ui.dropdown(
        options={
            "0 · swap only, any environment": 0,
            "1 · plus neighbouring atom": 1,
            "2 · two bonds of context": 2,
            "3 · three bonds of context": 3,
        },
        value="0 · swap only, any environment",
        label="Environment radius",
    )
    min_rule_pairs = mo.ui.slider(
        3, 30, value=5, step=1, debounce=True, label="Minimum pairs per transformation"
    )
    mo.hstack([rule_radius, min_rule_pairs], widths="equal")
    return min_rule_pairs, rule_radius


@app.cell(hide_code=True)
def _(database_path, math, min_rule_pairs, rule_radius, sqlite3, statistics):
    _sql = """
        SELECT DISTINCT
            p.rule_environment_id,
            fs.smiles,
            ts.smiles,
            ef.pseudosmiles,
            c1.public_id,
            c2.public_id,
            cp2.value - cp1.value
        FROM pair p
        JOIN compound c1 ON c1.id = p.compound1_id
        JOIN compound c2 ON c2.id = p.compound2_id
        JOIN rule_environment re ON re.id = p.rule_environment_id
        JOIN environment_fingerprint ef ON ef.id = re.environment_fingerprint_id
        JOIN rule r ON r.id = re.rule_id
        JOIN rule_smiles fs ON fs.id = r.from_smiles_id
        JOIN rule_smiles ts ON ts.id = r.to_smiles_id
        JOIN property_name pn ON pn.name = 'Combined_pIC50'
        JOIN compound_property cp1
            ON cp1.compound_id = c1.id AND cp1.property_name_id = pn.id
        JOIN compound_property cp2
            ON cp2.compound_id = c2.id AND cp2.property_name_id = pn.id
        WHERE re.radius = ?
    """
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as _connection:
        _database_rows = _connection.execute(_sql, (rule_radius.value,)).fetchall()

    # One delta per distinct compound pair: mmpdb can record the same pair
    # several times under one rule environment (e.g. symmetric attachment points).
    _by_environment = {}
    for _env_id, _from, _to, _environment, _c1, _c2, _delta in _database_rows:
        _entry = _by_environment.setdefault(
            _env_id, {"from": _from, "to": _to, "environment": _environment, "deltas": {}}
        )
        _entry["deltas"][(_c1, _c2)] = _delta


    def _sign_test_p(_k, _n):
        """Exact two-sided binomial sign test for k of n moves in one direction."""
        if _n == 0:
            return 1.0
        _tail = sum(math.comb(_n, _i) for _i in range(max(_k, _n - _k), _n + 1)) / 2**_n
        return min(1.0, 2 * _tail)


    recurring_rules = []
    rule_pair_deltas = {}
    for _env_id, _entry in _by_environment.items():
        _values = list(_entry["deltas"].values())
        if len(_values) < min_rule_pairs.value:
            continue
        _up = sum(_value > 0.1 for _value in _values)
        _down = sum(_value < -0.1 for _value in _values)
        _moved = _up + _down
        _quartiles = statistics.quantiles(_values, n=4)
        _row = {
            "rule_environment_id": _env_id,
            "from_fragment": _entry["from"],
            "to_fragment": _entry["to"],
            "pairs": len(_values),
            "median_delta": round(statistics.median(_values), 2),
            "q1_delta": round(_quartiles[0], 2),
            "q3_delta": round(_quartiles[2], 2),
            "up": _up,
            "down": _down,
            "flat": len(_values) - _moved,
            "agreement": round(max(_up, _down) / _moved, 2) if _moved else None,
            "sign_p": _sign_test_p(_up, _moved),
        }
        if rule_radius.value > 0:
            _row["environment"] = _entry["environment"]
        recurring_rules.append(_row)
        rule_pair_deltas[_env_id] = _entry["deltas"]

    # Benjamini–Hochberg q-values across every transformation shown.
    recurring_rules.sort(key=lambda _row: _row["sign_p"])
    _running = 1.0
    for _rank in range(len(recurring_rules), 0, -1):
        _row = recurring_rules[_rank - 1]
        _running = min(_running, _row["sign_p"] * len(recurring_rules) / _rank)
        _row["q"] = _running
    for _row in recurring_rules:
        _increases = _row["up"] > _row["down"]
        if _row["q"] < 0.05:
            _row["trend"] = "increases" if _increases else "decreases"
        elif _row["sign_p"] < 0.05:
            _row["trend"] = "leans up" if _increases else "leans down"
        else:
            _row["trend"] = "no consistent direction"
        _row["sign_p"] = float(f"{_row['sign_p']:.2g}")
        _row["q"] = float(f"{_row['q']:.2g}")
    recurring_rules.sort(key=lambda _row: (_row["q"], -_row["pairs"]))
    return (recurring_rules,)


@app.cell(hide_code=True)
def _(min_rule_pairs, mo, recurring_rules, rule_radius):
    recurring_table = mo.ui.table(
        recurring_rules,
        selection="single",
        page_size=8,
        show_download=True,
        label="Recurring transformations (Combined pIC50) · select one to inspect its pairs",
    )
    _trend_counts = {
        _trend: sum(_row["trend"] == _trend for _row in recurring_rules)
        for _trend in (
            "increases", "decreases", "leans up", "leans down", "no consistent direction"
        )
    }
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"**{len(recurring_rules)} transformations** at radius "
                    f"{rule_radius.value} with ≥ {min_rule_pairs.value} pairs: "
                    f"**{_trend_counts['increases']}** consistently increase inhibition, "
                    f"**{_trend_counts['decreases']}** consistently decrease it "
                    f"(q < 0.05); a further {_trend_counts['leans up']} lean up and "
                    f"{_trend_counts['leans down']} lean down (uncorrected p < 0.05); "
                    f"{_trend_counts['no consistent direction']} show no consistent direction."
                ),
                kind="info",
            ),
            recurring_table,
        ]
    )
    return (recurring_table,)


@app.cell(hide_code=True)
def _(
    database_path,
    environment_svg,
    fragment_reaction_svg,
    mo,
    recurring_rules,
    recurring_table,
    rule_radius,
    sqlite3,
):
    chosen_rule = (
        recurring_table.value[0]
        if recurring_table.value
        else (recurring_rules[0] if recurring_rules else None)
    )
    _sql = """
        SELECT DISTINCT
            c1.public_id,
            c2.public_id,
            c1.clean_smiles,
            c2.clean_smiles,
            combined1.value,
            combined2.value,
            openadmet1.value IS NOT NULL,
            openadmet2.value IS NOT NULL,
            chembl1.value IS NOT NULL,
            chembl2.value IS NOT NULL
        FROM pair p
        JOIN compound c1 ON c1.id = p.compound1_id
        JOIN compound c2 ON c2.id = p.compound2_id
        JOIN property_name combined ON combined.name = 'Combined_pIC50'
        JOIN property_name openadmet ON openadmet.name = 'OpenADMET_pIC50'
        JOIN property_name chembl ON chembl.name = 'ChEMBL_pIC50'
        JOIN compound_property combined1
            ON combined1.compound_id = c1.id AND combined1.property_name_id = combined.id
        JOIN compound_property combined2
            ON combined2.compound_id = c2.id AND combined2.property_name_id = combined.id
        LEFT JOIN compound_property openadmet1
            ON openadmet1.compound_id = c1.id AND openadmet1.property_name_id = openadmet.id
        LEFT JOIN compound_property openadmet2
            ON openadmet2.compound_id = c2.id AND openadmet2.property_name_id = openadmet.id
        LEFT JOIN compound_property chembl1
            ON chembl1.compound_id = c1.id AND chembl1.property_name_id = chembl.id
        LEFT JOIN compound_property chembl2
            ON chembl2.compound_id = c2.id AND chembl2.property_name_id = chembl.id
        WHERE p.rule_environment_id = ?
    """


    def _source(_in_openadmet, _in_chembl):
        return " + ".join(
            _name
            for _name, _present in (("OpenADMET", _in_openadmet), ("ChEMBL", _in_chembl))
            if _present
        )


    rule_pair_rows = []
    if chosen_rule is not None:
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as _connection:
            _database_pairs = _connection.execute(
                _sql, (chosen_rule["rule_environment_id"],)
            ).fetchall()
        for (
            _from_id,
            _to_id,
            _from_smiles,
            _to_smiles,
            _from_value,
            _to_value,
            _from_openadmet,
            _to_openadmet,
            _from_chembl,
            _to_chembl,
        ) in _database_pairs:
            rule_pair_rows.append(
                {
                    "from_id": _from_id,
                    "to_id": _to_id,
                    "from_pIC50": round(_from_value, 2),
                    "to_pIC50": round(_to_value, 2),
                    "delta_pIC50": round(_to_value - _from_value, 2),
                    "from_source": _source(_from_openadmet, _from_chembl),
                    "to_source": _source(_to_openadmet, _to_chembl),
                    "from_smiles": _from_smiles,
                    "to_smiles": _to_smiles,
                }
            )
        rule_pair_rows.sort(key=lambda _row: -_row["delta_pIC50"])

    rule_pair_table = mo.ui.table(
        rule_pair_rows,
        selection="single",
        page_size=6,
        show_download=True,
        label="Pairs behind the selected transformation",
    )
    if chosen_rule is None:
        _output = mo.callout("No transformation meets these settings.", kind="warn")
    else:
        _environment = chosen_rule.get("environment")
        _panels = [
            mo.vstack(
                [
                    mo.md("**Transformation**"),
                    mo.Html(
                        fragment_reaction_svg(
                            chosen_rule["from_fragment"], chosen_rule["to_fragment"]
                        )
                    ),
                ]
            )
        ]
        if _environment:
            _panels.append(
                mo.vstack(
                    [
                        mo.md(f"**Environment (radius {rule_radius.value})**"),
                        mo.Html(environment_svg(_environment)),
                        mo.md(
                            "R marks where the fragment is swapped; dotted bonds "
                            "continue into the rest of the molecule."
                        ).style({"font-size": "0.85em", "color": "#666"}),
                    ]
                )
            )
        _output = mo.vstack(
            [
                mo.hstack(_panels, justify="start", gap=2),
                mo.md(
                    f"{chosen_rule['pairs']} pairs · median ΔpIC50 "
                    f"{chosen_rule['median_delta']:+.2f} · "
                    f"{chosen_rule['up']} up, {chosen_rule['down']} down, "
                    f"{chosen_rule['flat']} flat · trend: *{chosen_rule['trend']}*"
                ),
                rule_pair_table,
            ]
        )
    _output
    return chosen_rule, rule_pair_rows, rule_pair_table


@app.cell(hide_code=True)
def _(
    aligned_pair_svgs,
    chosen_rule,
    measurement_by_id,
    mo,
    rule_pair_rows,
    rule_pair_table,
):
    _chosen = (
        rule_pair_table.value[0]
        if rule_pair_table.value
        else (rule_pair_rows[0] if rule_pair_rows else None)
    )
    if _chosen is None:
        _overview = mo.md("")
    else:
        _svgs = dict(
            zip(
                ("from", "to"),
                aligned_pair_svgs(
                    _chosen["from_smiles"],
                    _chosen["to_smiles"],
                    360,
                    190,
                    highlight_changes=True,
                ),
            )
        )
        _cards = []
        for _label, _side in (("A", "from"), ("B", "to")):
            _id = _chosen[f"{_side}_id"]
            _measurement = measurement_by_id.get(_id)
            _interval = (
                f", OpenADMET interval [{_measurement['conf_low']:.2f}, "
                f"{_measurement['conf_high']:.2f}]"
                if _measurement
                else ""
            )
            _cards.append(
                mo.vstack(
                    [
                        mo.md(f"**{_label} · {_id}**"),
                        mo.Html(_svgs[_side]),
                        mo.md(
                            f"Combined pIC50 **{_chosen[f'{_side}_pIC50']:.2f}**"
                            f" ({_chosen[f'{_side}_source']}){_interval}"
                        ),
                    ]
                )
            )
        _delta = _chosen["delta_pIC50"]
        _median = chosen_rule["median_delta"]
        if abs(_delta) <= 0.1:
            _direction, _kind = "leaves observed CYP3A4 inhibition essentially unchanged", "info"
        elif _delta > 0:
            _direction, _kind = f"increases observed CYP3A4 inhibition by **{_delta:.2f}**", "danger"
        else:
            _direction, _kind = f"decreases observed CYP3A4 inhibition by **{abs(_delta):.2f}**", "success"
        _overview = mo.vstack(
            [
                mo.hstack(_cards, widths="equal"),
                mo.callout(
                    mo.md(
                        f"**In this pair, A → B {_direction}.** Across all "
                        f"{chosen_rule['pairs']} pairs of this transformation the median "
                        f"ΔpIC50 is {_median:+.2f} "
                        f"({chosen_rule['up']} up, {chosen_rule['down']} down, "
                        f"{chosen_rule['flat']} flat)."
                    ),
                    kind=_kind,
                ),
            ]
        )
    _overview
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · What should I try on my own molecule?

    Pick an anchor and evidence source. **OpenADMET is the primary study view and
    is selected by default.** ChEMBL provides broader mixed-assay context.
    Combined is exploratory: it merges the two sources and may mix measurement
    contexts, so interpret its effect estimates cautiously.

    The custom widget applies database transformations, groups duplicate products,
    and exposes support, radius, direction, and source pairs. Try increasing
    minimum support or radius, then click a product to inspect the measurements
    behind its selected rule.
    """)
    return


@app.cell(hide_code=True)
def _(mo, queries):
    def _anchor_option_label(_query):
        _source_label = _query["property"]
        if _source_label == "OpenADMET_pIC50":
            _source_label = "OpenADMET_pIC50 (primary)"
        elif _source_label == "ChEMBL_pIC50":
            _source_label = "ChEMBL_pIC50 (broad assay context)"
        elif _source_label == "Combined_pIC50":
            _source_label = "Combined_pIC50 (exploratory mixed-source)"
        return f"{_query['id']} · {_source_label} · {_query['products']} products"


    _anchor_options = {_anchor_option_label(_query): _query for _query in queries}
    default_anchor = next(
        _query for _query in queries if _query["property"] == "OpenADMET_pIC50"
    )
    anchor_picker = mo.ui.dropdown(
        options=_anchor_options,
        value=_anchor_option_label(default_anchor),
        label="Anchor and evidence source",
    )
    anchor_picker
    return anchor_picker, default_anchor


@app.cell(hide_code=True)
def _(mo):
    max_nodes_input = mo.ui.number(
        start=5, stop=200, step=5, value=30, label="Max products shown"
    )
    mo.hstack([max_nodes_input], justify="start", gap=2)
    return (max_nodes_input,)


@app.cell(hide_code=True)
def _(
    TransformDataset,
    TransformGraph,
    database_path,
    default_anchor,
    mmp_dir,
    mo,
):
    def load_transform_data(query):
        """Load one anchor's transform predictions with source-pair provenance."""
        return TransformDataset.from_tsv(
            mmp_dir / query["file"],
            original_smiles=query["smiles"],
            mmpdb_path=database_path,
        )


    # Built once: anchor and product-limit changes go through `update` in the next
    # cell, so the user's in-widget filters, direction, and selection survive.
    transform_graph = TransformGraph(
        load_transform_data(default_anchor), height=720, direction="lower"
    )
    graph = mo.ui.anywidget(transform_graph)
    graph
    return graph, load_transform_data, transform_graph


@app.cell(hide_code=True)
def _(anchor_picker, load_transform_data, max_nodes_input, transform_graph):
    anchor = anchor_picker.value
    transform_data = load_transform_data(anchor)
    # Keep the user's widget filters; fall back to defaults if the new dataset
    # rejects them.
    try:
        transform_graph.update(
            transform_data,
            filters=transform_graph.state.filters,
            max_nodes=max_nodes_input.value,
        )
    except ValueError:
        transform_graph.update(transform_data, max_nodes=max_nodes_input.value)
    return anchor, transform_data


@app.cell(hide_code=True)
def _(Chem, Draw, Point2D, rdChemReactions, rdDepictor, rdFMCS):
    def aligned_pair_svgs(
        reference_smiles, smiles, width=360, height=190, highlight_changes=False
    ):
        """Depict two molecules with their maximum common substructure in the same pose.

        The reference gets standard 2D coordinates; the second molecule's MCS atoms
        are pinned to the matching reference positions so the shared scaffold lines up.
        Falls back to independent layouts when no MCS is found within the timeout.
        With `highlight_changes`, atoms outside the common substructure are highlighted.
        """
        _reference = Chem.MolFromSmiles(reference_smiles)
        _molecule = Chem.MolFromSmiles(smiles)
        rdDepictor.Compute2DCoords(_reference)
        _mcs = rdFMCS.FindMCS(
            [_reference, _molecule],
            timeout=2,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
        )
        _core = (
            Chem.MolFromSmarts(_mcs.smartsString)
            if _mcs.smartsString and not _mcs.canceled
            else None
        )
        _reference_match = _reference.GetSubstructMatch(_core) if _core else ()
        _molecule_match = _molecule.GetSubstructMatch(_core) if _core else ()
        _coord_map = {}
        if _reference_match and _molecule_match:
            _conformer = _reference.GetConformer()
            for _reference_index, _molecule_index in zip(
                _reference_match, _molecule_match
            ):
                _position = _conformer.GetAtomPosition(_reference_index)
                _coord_map[_molecule_index] = Point2D(_position.x, _position.y)
        rdDepictor.Compute2DCoords(
            _molecule, canonOrient=not _coord_map, coordMap=_coord_map
        )

        def _draw(_mol, _match):
            _drawer = Draw.MolDraw2DSVG(width, height)
            _options = _drawer.drawOptions()
            _options.clearBackground = False
            _options.padding = 0.08
            _options.baseFontSize = 0.56
            _options.fixedBondLength = 24
            _changed = (
                [_index for _index in range(_mol.GetNumAtoms()) if _index not in _match]
                if highlight_changes and _match
                else []
            )
            _drawer.DrawMolecule(
                _mol,
                highlightAtoms=_changed,
                highlightAtomColors={_index: (0.88, 0.56, 0.12, 0.45) for _index in _changed},
                highlightBonds=[],
            )
            _drawer.FinishDrawing()
            return _drawer.GetDrawingText()

        return _draw(_reference, set(_reference_match)), _draw(
            _molecule, set(_molecule_match)
        )


    def fragment_reaction_svg(from_fragment, to_fragment, width=300, height=160):
        """Draw a matched-pair transformation as a from-fragment >> to-fragment reaction."""
        _reaction = rdChemReactions.ReactionFromSmarts(
            f"{from_fragment}>>{to_fragment}",
            useSmiles=True,
        )
        return Draw.ReactionToImage(
            _reaction,
            subImgSize=(width, height),
            useSVG=True,
        )


    def environment_svg(pseudosmiles, width=320, height=180):
        """Draw an mmpdb environment fingerprint around the attachment point.

        The attachment point is labelled "R" and highlighted; unspecified
        neighbours (`~*`) are drawn as unlabelled dotted stubs.
        """
        _mol = Chem.MolFromSmarts(pseudosmiles)
        _mol.UpdatePropertyCache(strict=False)
        rdDepictor.Compute2DCoords(_mol)
        _attachment = []
        for _atom in _mol.GetAtoms():
            if _atom.GetAtomicNum() != 0:
                continue
            if _atom.GetAtomMapNum() == 1:
                _atom.SetAtomMapNum(0)
                _atom.SetProp("atomLabel", "R")
                _attachment.append(_atom.GetIdx())
            else:
                _atom.SetProp("atomLabel", " ")
        _drawer = Draw.MolDraw2DSVG(width, height)
        _options = _drawer.drawOptions()
        _options.clearBackground = False
        _options.padding = 0.12
        _options.fixedBondLength = 28
        _drawer.DrawMolecule(
            _mol,
            highlightAtoms=_attachment,
            highlightAtomColors={_index: (1.0, 0.8, 0.4) for _index in _attachment},
            highlightBonds=[],
        )
        _drawer.FinishDrawing()
        return _drawer.GetDrawingText()

    return aligned_pair_svgs, environment_svg, fragment_reaction_svg


@app.cell(hide_code=True)
def _(aligned_pair_svgs, fragment_reaction_svg, measurement_by_id, mo):
    def render_matched_pair(pair):
        """Show both compounds, the direction of change, and the transformation for one matched pair."""
        _a = measurement_by_id[pair["from_id"]]
        _b = measurement_by_id[pair["to_id"]]
        _svgs = aligned_pair_svgs(
            _a["smiles"], _b["smiles"], 360, 190, highlight_changes=True
        )
        _cards = []
        for _label, _compound, _svg in (("A", _a, _svgs[0]), ("B", _b, _svgs[1])):
            _cards.append(
                mo.vstack(
                    [
                        mo.md(f"**{_label} · {_compound['id']}**"),
                        mo.Html(_svg),
                        mo.md(
                            f"pIC50 **{_compound['pIC50']:.2f}**, interval "
                            f"[{_compound['conf_low']:.2f}, {_compound['conf_high']:.2f}]"
                        ),
                    ]
                )
            )

        _reaction_svg = fragment_reaction_svg(
            pair["from_fragment"], pair["to_fragment"]
        )

        _delta = pair["delta_pIC50"]
        _interval_note = (
            "The reported A/B activity intervals do not overlap."
            if pair["interval_separated"]
            else (
                "The reported A/B activity intervals overlap, so the direction is "
                "not robust to the interval endpoints."
            )
        )
        if _delta > 0:
            _direction_callout = mo.callout(
                mo.md(
                    f"**A → B increases observed CYP3A4 inhibition.** "
                    f"pIC50 rises by **{_delta:.2f}**. {_interval_note}"
                ),
                kind="danger",
            )
        elif _delta < 0:
            _direction_callout = mo.callout(
                mo.md(
                    f"**A → B decreases observed CYP3A4 inhibition.** "
                    f"pIC50 falls by **{abs(_delta):.2f}**. {_interval_note}"
                ),
                kind="success",
            )
        else:
            _direction_callout = mo.callout(
                mo.md(
                    f"**A → B shows no change in observed CYP3A4 inhibition.** "
                    f"{_interval_note}"
                ),
                kind="info",
            )

        return mo.vstack(
            [
                mo.hstack(_cards, widths="equal"),
                _direction_callout,
                mo.md("**Matched-pair transformation**"),
                mo.Html(_reaction_svg),
                mo.md(
                    f"**B − A = {pair['delta_pIC50']:+.2f} pIC50** · "
                    f"endpoint bounds [{pair['delta_bound_low']:+.2f}, "
                    f"{pair['delta_bound_high']:+.2f}]"
                ),
            ]
        )


    return (render_matched_pair,)


@app.cell(hide_code=True)
def _(Draw, aligned_pair_svgs, anchor, base64, graph, mo, rdChemReactions):
    graph_state = graph.state
    _compound = graph_state.selected_compound
    _stats = graph_state.selected_stats


    def _fmt(_value, _spec):
        return "—" if _value is None else format(_value, _spec)


    def _scaled_svg(_svg):
        """Embed an SVG as an image that shrinks to fit its column."""
        _encoded = base64.b64encode(_svg.encode()).decode()
        return mo.Html(
            f'<img src="data:image/svg+xml;base64,{_encoded}" '
            'style="max-width:100%;height:auto;display:block;margin:0 auto">'
        )


    if _compound is None or _stats is None:
        _summary = mo.callout("Click a product in the graph to see its summary.", kind="neutral")
    else:
        _row = next(
            (_r for _r in graph_state.rows() if _r["ID"] == _compound.id), {}
        )
        _reaction = rdChemReactions.ReactionFromSmarts(
            f"{_stats.from_smiles}>>{_stats.to_smiles}", useSmiles=True
        )
        _reaction_svg = Draw.ReactionToImage(
            _reaction, subImgSize=(150, 110), useSVG=True
        )
        _anchor_svg, _product_svg = aligned_pair_svgs(anchor["smiles"], _compound.smiles)
        _estimate = anchor["pIC50"] + _stats.median
        _cards = mo.hstack(
            [
                mo.vstack(
                    [
                        mo.md(f"**Anchor · {anchor['id']}**"),
                        _scaled_svg(_anchor_svg),
                        mo.md(f"Measured pIC50 **{anchor['pIC50']:.2f}**"),
                    ]
                ),
                mo.vstack(
                    [
                        mo.md("**Rule applied**"),
                        _scaled_svg(_reaction_svg),
                    ],
                    align="center",
                ),
                mo.vstack(
                    [
                        mo.md(f"**Product · {_compound.id}**"),
                        _scaled_svg(_product_svg),
                    ]
                ),
            ],
            widths=[1, 0.8, 1],
            align="center",
        )
        _tiles = mo.hstack(
            [
                mo.stat(
                    value=str(_stats.count),
                    label="Supporting pairs",
                    caption=f"{_row.get('evidence', '—')} evidence",
                    bordered=True,
                ),
                mo.stat(
                    value=_fmt(_stats.median, "+.2f"),
                    label="Median ΔpIC50",
                    caption=(
                        f"IQR {_fmt(_stats.q1, '+.2f')} … {_fmt(_stats.q3, '+.2f')}"
                    ),
                    bordered=True,
                ),
                mo.stat(
                    value=f"{_fmt(_stats.avg, '+.2f')} ± {_fmt(_stats.std, '.2f')}",
                    label="Mean ± std",
                    caption=(
                        f"range {_fmt(_stats.min, '+.2f')} … {_fmt(_stats.max, '+.2f')}"
                    ),
                    bordered=True,
                ),
                mo.stat(
                    value=_fmt(_stats.p_value, ".3f"),
                    label="p-value",
                    caption="",
                    bordered=True,
                ),
                mo.stat(
                    value=f"{_estimate:.2f}",
                    label="Estimated product pIC50",
                    caption="anchor pIC50 + median Δ",
                    bordered=True,
                ),
            ],
            widths="equal",
        )
        _consistent = (_stats.min is not None and _stats.min > 0) or (
            _stats.max is not None and _stats.max < 0
        )
        _consistency_note = (
            "Every supporting pair moves in this direction."
            if _consistent
            else "Supporting pairs disagree on direction, so treat this as a tendency."
        )
        if _stats.median > 0:
            _verdict = mo.callout(
                mo.md(
                    f"**This rule tends to increase CYP3A4 inhibition** "
                    f"(median ΔpIC50 {_stats.median:+.2f}). {_consistency_note}"
                ),
                kind="danger",
            )
        elif _stats.median < 0:
            _verdict = mo.callout(
                mo.md(
                    f"**This rule tends to decrease CYP3A4 inhibition** "
                    f"(median ΔpIC50 {_stats.median:+.2f}). {_consistency_note}"
                ),
                kind="success",
            )
        else:
            _verdict = mo.callout(
                mo.md(f"**This rule shows no median change.** {_consistency_note}"),
                kind="info",
            )
        _summary = mo.vstack(
            [
                _cards,
                _tiles,
                _verdict,
            ]
        )
    mo.vstack([mo.md("### Selected product and the rule behind it"), _summary])
    return (graph_state,)


@app.cell(hide_code=True)
def _(anchor, graph_state, mo, pd):
    import dataclasses
    source_evidence = pd.DataFrame([dataclasses.asdict(p) for p in graph_state.source_pairs()])
    mo.vstack(
        [
            mo.md(f"### Measured source pairs for `{anchor['property']}`"),
            mo.ui.table(
                source_evidence,
                selection=None,
                page_size=7,
                show_download=True,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(anchor, mo, transform_data):
    _property = anchor["property"]
    _suggestions = []
    for _record in transform_data.records:
        _stats = _record.properties.get(_property)
        if _stats is None or _stats.median is None or _stats.median >= 0:
            continue
        _suggestions.append(
            {
                "product_smiles": _record.smiles,
                "from_fragment": _stats.from_smiles,
                "to_fragment": _stats.to_smiles,
                "radius": _stats.radius,
                "pairs": _stats.count,
                "median_delta": round(_stats.median, 2),
                "q1_delta": round(_stats.q1, 2) if _stats.q1 is not None else None,
                "q3_delta": round(_stats.q3, 2) if _stats.q3 is not None else None,
                "all_pairs_decrease": _stats.max is not None and _stats.max < 0,
                "estimated_pIC50": round(anchor["pIC50"] + _stats.median, 2),
            }
        )
    _suggestions.sort(key=lambda _row: (-_row["pairs"], _row["median_delta"]))
    _strong = [
        _row
        for _row in _suggestions
        if _row["pairs"] >= transform_data.evidence_thresholds.strong
    ]
    _strong_consistent = [_row for _row in _strong if _row["all_pairs_decrease"]]
    mo.vstack(
        [
            mo.md(
                f"### Edits predicted to lower inhibition for {anchor['id']}\n\n"
                "Products from the transformations above whose supporting pairs have a "
                "negative median ΔpIC50, ranked by how many pairs support them. Prefer "
                "rows with many pairs that all agree, and treat single-pair rows as "
                "hypotheses to test."
            ),
            mo.ui.table(_suggestions, selection=None, page_size=6, show_download=True),
            mo.callout(
                mo.md(
                    f"**Answer 3 · For {anchor['id']} ({_property}), "
                    f"{len(_suggestions)} of {len(transform_data.records)} proposed "
                    f"products are predicted to lower CYP3A4 inhibition.** "
                    f"{len(_strong)} rest on ≥ {transform_data.evidence_thresholds.strong} "
                    f"pairs, and {len(_strong_consistent)} of those have every supporting "
                    "pair moving down. Because effects are context-dependent (section 2), "
                    "the most useful evidence is local: transformations observed close to "
                    "your scaffold, ideally at radius ≥ 1, from a single consistent assay."
                ),
                kind="success",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, robust_cliffs):
    mo.md(f"""
    ## 4 · What a chemist can learn about CYP3A4 from matched pairs

    1. **Small edits can matter a lot.** {len(robust_cliffs)} OpenADMET pairs change
       CYP3A4 inhibition by more than 10-fold with tight, non-overlapping
       measurement intervals. When a series inhibits CYP3A4, a single substitution
       can be enough to fix it.
    2. **There is no universal "anti-CYP3A4" swap.** Most recurring transformations
       have a median effect well under 2-fold, and almost none move inhibition
       consistently across molecules. At radius 0, methyl → H, F → H and Cl → H,
       the edits a chemist reaches for first, have median effects within ±0.05 log
       units across dozens to hundreds of pairs; Br → H is a rare exception that
       leans towards lower inhibition.
    3. **Use local, high-quality evidence.** Look for matched pairs on or near
       your own scaffold (radius ≥ 1), check how many pairs support a suggestion and
       whether they agree, and treat each one as a hypothesis to test.

    **Caveats.** Pairs come from single cuts with a changed fragment of at most 10
    heavy atoms and 40% of the molecule, so core and linker changes are not covered.
    ChEMBL values come from many assays, often sit at assay cut-offs (for example
    20 µM), and have no reported uncertainty. Statistical tests treat pairs as
    independent although pairs sharing a compound are not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Disclaimer

    LLM (with human review, where possible) were used to prepare large parts of the supporting code. `marimo-mmp` was mostly built using **GPT-5.6-Sol** and reviewed with **Opus 5.5**. Experiment and notebook structure and concept are human-made.
    """)
    return


if __name__ == "__main__":
    app.run()
