# Code for "The cognitive advantage of older internet users largely precedes going online"

This archive contains the code that builds the cohort panels, constructs the sequential target trials and reproduces every estimate, table and figure in the manuscript, Extended Data and Supplementary Information (version 4.13). An archived release with a DOI will be created on acceptance.

## Data

Individual-level data are not included and cannot be redistributed. They are available from the study teams on registration:

- HRS (RAND HRS and Gateway Harmonized HRS): https://hrs.isr.umich.edu
- ELSA (Harmonized ELSA): UK Data Service
- SHARE (easySHARE and wave modules): https://share-eric.eu
- CHARLS (Harmonized CHARLS and raw files for 2020 and 2021–2023): https://charls.pku.edu.cn
- LASI wave 1: https://www.iipsindia.ac.in/lasi

Pooled and cohort-level estimates underlying every figure are provided in the Source Data workbook; all analysis outputs (`results_*.csv`, `simulation_grid4.csv`) can be regenerated with the scripts below.

## Requirements

Python ≥3.11 with pandas, NumPy, SciPy, scikit-learn and matplotlib (figures); Fig. 1 is rendered from HTML with Playwright and Chrome. Set `OMP_NUM_THREADS=1` when running the AIPW scripts in parallel. Random seeds are fixed, so reruns reproduce the reported numbers.

## Structure

| Folder | Contents |
|---|---|
| `01_data_preparation` | Panel builders for HRS, ELSA, SHARE, CHARLS (including `build_charls_w6.py` and `charls2020_extra.py` for 2020 and 2021–2023) and LASI; additional covariates and outcomes; `build_trials.py` (sequential target trials). |
| `02_analysis` | `aipw.py` (cross-fitted augmented inverse probability weighting with gradient-boosted nuisance models, overlap trimming at 0.025, missing-outcome weights, person-clustered influence-function standard errors). Designs: `run_main.py` (target trials), `placebo.py`/`placebo2.py` (placebo trials), `pretrend.py`, `within*.py` (early versus late adoption), `es.py`/`es_stop.py` (event studies), `negctrl*.py` (negative-control outcomes), `newout.py`, `dementia.py`, `subgrp.py`/`hte*.py` (subgroups), `anatomy.py`/`stop_pred.py`/`gradient.py` (selection), `behav.py`/`vig_pred.py`/`es_stopvig.py` (physical activity), `simulate4.py` (simulation), `repool.py`/`pool_hk.py` (Hartung–Knapp–Sidik–Jonkman pooling), `corr_designs.py` (between-design correlations), `sitables*.py` (Supplementary tables). `charls_only_rerun` reruns the CHARLS trials only and merges them with the stored results of the other cohorts. |
| `03_figures` | `nh3fig.py` (Figs 2–8, Extended Data Figs 1–4 and Supplementary Figs 4–5), `svgfigs.py` (editable SVG versions), `gen_fig1.py` and `render_fig1.py` (Fig. 1), `fig_flow.py` (participant flow), `fig_dag3.py` with `figures.py` (causal diagram), `ncfig.py`, `ncfig6.py` and `nhfig.py` (shared plotting helpers). |
| `03_figures/sourcedata411.py` | Rebuilds the Source Data workbook from the plotted figure objects. |
| `resources` | Natural Earth 1:110m country boundaries for the map in Fig. 1. |

## Order of execution

1. Panels: `01_data_preparation/01_hrs_panel.py` … `06_lasi_cross.py`, `07_new_outcomes.py`, `build_charls_w6.py`.
2. Trials: `build_trials.py`.
3. Primary analysis: `USE_LAG=1 USE_EXTRA=1 python run_main.py hgb 0.025`; other designs as listed above.
4. Pooling: `repool.py`, `pool_hk.py`.
5. Figures and tables: `nh3fig.py fig2 fig3 fig4 fig5 fig6 fig7n fig8 ed1 ed2 ed3 ed4`, `gen_fig1.py` then `render_fig1.py`, `sitables*.py`, `sourcedata411.py`.

Some data paths and comments in the data-preparation scripts are in Chinese because the analysis was run on a Chinese-language system; variable names and the analysis code are in English. Paths to the raw survey files and to the derived analysis files are written as absolute paths at the top of each script and need to be edited to the local location of the data before running.

## Code style

Python code is formatted with black (line length 100); `ruff check` passes for syntax and import-style rules.
