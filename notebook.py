import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full", app_title="CYP3A4 matched-pair explorer")


@app.cell(hide_code=True)
def _():
    import csv
    import json
    import sqlite3
    import statistics
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw, rdChemReactions

    from marimo_mmp import TransformDataset, TransformGraph
    from marimo_mmp.depiction import molecule_svg

    return (
        Draw,
        Path,
        TransformDataset,
        TransformGraph,
        csv,
        json,
        mo,
        molecule_svg,
        plt,
        rdChemReactions,
        sqlite3,
        statistics,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CYP3A4 matched-pair explorer

    A small structural edit can move CYP3A4 inhibition by a full log unit — or
    appear to, when the measurements are broad. This goal of this notebook is to:

    1. inspect measurement uncertainty;
    2. find matched-pair activity cliffs;
    3. see whether recurring edits keep the same sign;
    4. explore proposed transformations with the custom **marimo-mmp** widget.

    **pIC50 = −log10(IC50 in mol/L)**, so higher values mean stronger inhibition.

    ChEMBL values use geometric-mean IC50 aggregation
    within and across assays. For structures present in both
    sources, the combined value is their geometric-mean IC50.
    """)
    return


@app.cell
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


@app.cell
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
    ## 1 · Any activity cliffs?

    The database contains **{unique_molecular_pairs} distinct OpenADMET molecular
    pairs** under the chosen fragmentation settings. A cliff is an observed pair
    whose absolute activity change exceeds the selected threshold.

    The interval bounds shown below combine the reported endpoints:
    `[B_low − A_high, B_high − A_low]`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cliff_threshold = mo.ui.slider(
        0.3, 2.5, value=1.0, step=0.1, label="Minimum |ΔpIC50|"
    )
    width_threshold = mo.ui.slider(
        0.2, 4.0, value=3.0, step=0.1, label="Maximum interval width"
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
    return cliffs, eligible_pairs


@app.cell(hide_code=True)
def _(cliffs, eligible_pairs, mo, pair_rows):
    mo.callout(
        mo.md(
            f"**{len(cliffs)} candidate cliffs** among "
            f"**{len(eligible_pairs)} rule pairs** "
            f"({len(pair_rows)} before filtering).Try changing the ranges to see how it will affect number of outliers (cliffs)"
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(
    cliff_threshold,
    cliffs,
    eligible_pairs,
    measurements,
    plt,
    width_threshold,
):
    cliff_figure, _axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    _axes[0].hist(
        [_row["ci_width"] for _row in measurements],
        bins=30,
        color="#287d8e",
        edgecolor="white",
    )
    _axes[0].set(
        xlabel="Reported interval width (pIC50)",
        ylabel="Molecules",
        title="Measurement precision varies widely",
    )
    _axes[0].axvline(
        width_threshold.value,
        color="#bb4b31",
        linestyle="--",
        linewidth=1.5,
        label=f"maximum = {width_threshold.value:.1f}",
    )
    _axes[0].legend(frameon=False)

    _cliff_keys = {
        (_row["rule_id"], _row["from_id"], _row["to_id"]) for _row in cliffs
    }
    for _is_cliff, _color, _label in (
        (False, "#a7b1b7", "below threshold"),
        (True, "#bb4b31", "candidate cliff"),
    ):
        _subset = [
            _row
            for _row in eligible_pairs
            if ((_row["rule_id"], _row["from_id"], _row["to_id"]) in _cliff_keys)
            == _is_cliff
        ]
        _axes[1].scatter(
            [_row["from_pIC50"] for _row in _subset],
            [_row["to_pIC50"] for _row in _subset],
            s=24 if _is_cliff else 16,
            alpha=0.75 if _is_cliff else 0.35,
            color=_color,
            label=_label,
        )
    _limits = [1.0, 7.6]
    _axes[1].plot(_limits, _limits, color="#555555", linewidth=0.8)
    _axes[1].plot(
        _limits,
        [_value + cliff_threshold.value for _value in _limits],
        color="#bb4b31",
        linestyle="--",
        linewidth=0.8,
    )
    _axes[1].plot(
        _limits,
        [_value - cliff_threshold.value for _value in _limits],
        color="#bb4b31",
        linestyle="--",
        linewidth=0.8,
    )
    _axes[1].set(
        xlabel="From pIC50",
        ylabel="To pIC50",
        title="Matched-pair activity changes",
        xlim=_limits,
        ylim=_limits,
    )
    _axes[1].legend(frameon=False)
    cliff_figure
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
def _(
    Draw,
    cliff_table,
    cliffs,
    measurement_by_id,
    mo,
    molecule_svg,
    rdChemReactions,
):
    chosen_pair = (
        cliff_table.value[0]
        if cliff_table.value
        else (cliffs[0] if cliffs else None)
    )
    if chosen_pair is None:
        pair_display = mo.callout("No pair meets these settings.", kind="warn")
    else:
        _a = measurement_by_id[chosen_pair["from_id"]]
        _b = measurement_by_id[chosen_pair["to_id"]]
        _cards = []
        for _label, _compound in (("A", _a), ("B", _b)):
            _cards.append(
                mo.vstack(
                    [
                        mo.md(f"**{_label} · {_compound['id']}**"),
                        mo.Html(molecule_svg(_compound["smiles"], None, 360, 190)),
                        mo.md(
                            f"pIC50 **{_compound['pIC50']:.2f}**, interval "
                            f"[{_compound['conf_low']:.2f}, {_compound['conf_high']:.2f}]"
                        ),
                    ]
                )
            )

        _reaction_smiles = (
            f"{chosen_pair['from_fragment']}>>{chosen_pair['to_fragment']}"
        )
        _reaction = rdChemReactions.ReactionFromSmarts(
            _reaction_smiles,
            useSmiles=True,
        )
        _reaction_svg = Draw.ReactionToImage(
            _reaction,
            subImgSize=(300, 160),
            useSVG=True,
        )

        _delta = chosen_pair["delta_pIC50"]
        _interval_note = (
            "The reported A/B activity intervals do not overlap."
            if chosen_pair["interval_separated"]
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

        pair_display = mo.vstack(
            [
                mo.hstack(_cards, widths="equal"),
                _direction_callout,
                mo.md("**Matched-pair transformation**"),
                mo.Html(_reaction_svg),
                mo.md(
                    f"**B − A = {chosen_pair['delta_pIC50']:+.2f} pIC50** · "
                    f"endpoint bounds [{chosen_pair['delta_bound_low']:+.2f}, "
                    f"{chosen_pair['delta_bound_high']:+.2f}]"
                ),
            ]
        )
    pair_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · Do recurring pair move the property in the same direction?

    Radius-zero transformation pool are aggregated here
    """)
    return


@app.cell
def _(pair_rows, statistics):
    _by_rule = {}
    for _row in pair_rows:
        _by_rule.setdefault(_row["rule_id"], []).append(_row)
    recurring_rules = []
    for _rule_id, _rows in _by_rule.items():
        _unique = {
            (_row["from_id"], _row["to_id"]): _row for _row in _rows
        }
        _values = [_row["delta_pIC50"] for _row in _unique.values()]
        if len(_values) < 3:
            continue
        recurring_rules.append(
            {
                "from_fragment": _rows[0]["from_fragment"],
                "to_fragment": _rows[0]["to_fragment"],
                "pairs": len(_values),
                "median_delta": round(statistics.median(_values), 3),
                "min_delta": round(min(_values), 3),
                "max_delta": round(max(_values), 3),
                "same_direction": min(_values) < 0 < max(_values),
            }
        )
    recurring_rules.sort(key=lambda _row: (-_row["pairs"], _row["from_fragment"]))
    return (recurring_rules,)


@app.cell(hide_code=True)
def _(mo, recurring_rules):
    mo.ui.table(
        recurring_rules,
        selection=None,
        page_size=8,
        show_download=True,
        label="Recurring radius-zero transformations",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Explore transformations and inspect their evidence

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
    _default_anchor = next(
        _query for _query in queries if _query["property"] == "OpenADMET_pIC50"
    )
    anchor_picker = mo.ui.dropdown(
        options=_anchor_options,
        value=_anchor_option_label(_default_anchor),
        label="Anchor and evidence source",
    )
    anchor_picker
    return (anchor_picker,)


@app.cell(hide_code=True)
def _(TransformDataset, TransformGraph, anchor_picker, mmp_dir, mo):
    anchor = anchor_picker.value
    transform_data = TransformDataset.from_tsv(
        mmp_dir / anchor["file"],
        original_smiles=anchor["smiles"],
        mmpdb_path=mmp_dir / "cyp3a4.mmpdb",
    )
    graph = mo.ui.anywidget(TransformGraph(transform_data, max_nodes=45, height=720))
    graph
    return anchor, graph


@app.cell(hide_code=True)
def _(anchor, graph, mo):
    graph_state = graph.state
    source_evidence = []
    for _pair in graph_state.source_pairs():
        if _pair.from_value is None or _pair.to_value is None:
            continue
        source_evidence.append(
            {
                "from_id": _pair.from_id,
                "to_id": _pair.to_id,
                "from_source": "OpenADMET" if _pair.from_id.startswith("OCNT-") else "ChEMBL",
                "to_source": "OpenADMET" if _pair.to_id.startswith("OCNT-") else "ChEMBL",
                "from_pIC50": _pair.from_value,
                "to_pIC50": _pair.to_value,
                "observed_delta": _pair.delta,
                "includes_anchor": anchor["id"] in (_pair.from_id, _pair.to_id),
            }
        )
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


if __name__ == "__main__":
    app.run()
