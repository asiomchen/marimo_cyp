# CYP3A4 matched-pair EDA

This study is a compact, reproducible exploration of CYP3A4 inhibition. It
combines the OpenADMET challenge training measurements with a narrowly filtered
ChEMBL CYP3A4 subset, identifies matched molecular pairs, and uses the custom
`marimo-mmp` widget to inspect transformations and their experimental evidence.

The notebook focuses on:

- uncertainty-aware activity cliffs in the OpenADMET measurements;
- recurring structural edits and the consistency of their activity changes;
- comparison of OpenADMET, ChEMBL, and combined transformation statistics;
- interactive inspection of products and source pairs with `TransformGraph`.

## Reproduce the study

From this directory, create the locked environment:

```bash
uv sync --locked
```

There are exactly two data-generation stages:

```bash
# Networked: refresh and pin the two original source tables.
uv run --locked python pull_sources.py --refresh

# Offline: curate the sources and generate the MMPDB and transform files.
uv run --locked python generate_mmp.py
```

`pull_sources.py` uses the official `chembl-webresource-client` and requests
only exact IC50 measurements for CYP3A4 (`CHEMBL340`) in nM from direct,
high-confidence assays. It also downloads the OpenADMET training CSV at a
pinned repository revision. The resulting hashes, source revision, ChEMBL
release, filters, aggregation method, and row counts are recorded in
`data/source/manifest.json`.

Without `--refresh`, `pull_sources.py` makes no network requests and only
verifies the checked-in source files against that manifest:

```bash
uv run --locked python pull_sources.py
```

`generate_mmp.py` verifies those hashes, canonicalizes and curates the source
structures, constructs one MMPDB with `OpenADMET_pIC50`, `ChEMBL_pIC50`, and
`Combined_pIC50`, and generates the nine transform tables listed in
`data/mmp/queries.json`. Its database metadata date is fixed to 2026-09-29 so
rebuilds do not acquire an arbitrary timestamp.

Repeated ChEMBL IC50 measurements are combined by geometric mean within each
assay, then the assay-level IC50 values are combined by geometric mean for each
structure. When a canonical structure occurs in both sources, `Combined_pIC50`
is the pIC50 corresponding to the geometric mean of its OpenADMET and ChEMBL
IC50 values; otherwise it is the available source value. Equivalently, these
geometric means in IC50 space are arithmetic means in pIC50 space. The
source-specific properties remain separate in the database.

The notebook defaults to `OpenADMET_pIC50`, and its activity-cliff analysis
uses those more controlled measurements and their reported uncertainty
intervals. `ChEMBL_pIC50` provides broad, mixed-assay context.
`Combined_pIC50` is an exploratory mixed-source view rather than the primary
effect estimate, because it can merge different measurement contexts.

## Run the notebook

```bash
uv run --locked marimo edit notebook.py
```

The notebook reads only:

- `data/source/openadmet_train.csv`, the original input used for confidence
  intervals;
- `data/mmp/cyp3a4.mmpdb`, queried directly for matched pairs and properties;
- `data/mmp/queries.json` and the referenced `data/mmp/transforms_*.tsv` files;
- `data/mmp/manifest.json`, used for displayed build counts.

It does not depend on intermediate compound, pair, or rule tables. The ChEMBL
source snapshot is an input to `generate_mmp.py`, but the notebook does not need
to read it directly.

The local wheel in `distribution/` provides the widget and bundled frontend
assets used by the locked environment.

## Validate

```bash
uv run --locked marimo check --strict notebook.py
uv run --locked marimo export html notebook.py -o /tmp/cyp3a4-eda.html
```
