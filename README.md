# internet-cognition-selection

Analysis code for "The cognitive advantage of older internet users largely precedes going online".

The scripts build harmonized panels from HRS, ELSA, SHARE, CHARLS and LASI, set up sequential target trials, run the target-trial, placebo, pre-trend, early-versus-late and event-study analyses, and produce the figures and tables in the paper.

## Data

The survey data are not included. They can be obtained from the study teams after registration:

- HRS (RAND HRS, Gateway Harmonized HRS): https://hrs.isr.umich.edu
- ELSA (Harmonized ELSA): UK Data Service
- SHARE (easySHARE and wave modules): https://share-eric.eu
- CHARLS (Harmonized CHARLS; raw files for 2020 and 2021–2023): https://charls.pku.edu.cn
- LASI wave 1: https://www.iipsindia.ac.in/lasi

File paths are set at the top of each script; change them to where the data are stored locally.

## Requirements

Python 3.11 or later with pandas, NumPy, SciPy, scikit-learn and matplotlib. Fig. 1 is drawn as HTML and exported with Playwright. When running several estimation scripts at once, set `OMP_NUM_THREADS=1`.

## Contents

`01_data_preparation`
- `01_hrs_panel.py` to `06_lasi_cross.py`: cohort panels
- `07_new_outcomes.py`, `05_extra_covariates.py`: additional outcomes and covariates
- `build_charls_w6.py`, `charls2020_extra.py`: CHARLS 2020 and 2021–2023
- `build_trials.py`: sequential target trials

`02_analysis`
- `aipw.py`: cross-fitted AIPW estimator with gradient-boosted nuisance models
- `run_main.py`: target trials
- `placebo.py`, `placebo2.py`: placebo trials
- `pretrend.py`: differences before anyone starts
- `within*.py`: early versus late adopters
- `es.py`, `es_stop.py`: event studies around starting and stopping
- `negctrl*.py`, `newout.py`, `dementia.py`: other outcomes
- `subgrp.py`, `hte*.py`: subgroups
- `anatomy.py`, `stop_pred.py`, `gradient.py`: who starts and stops
- `behav.py`, `vig_pred.py`, `es_stopvig.py`: physical activity
- `simulate4.py`: simulation
- `repool.py`, `pool_hk.py`, `corr_designs.py`: pooling across cohorts
- `sitables*.py`: supplementary tables
- `charls_only_rerun/`: reruns CHARLS and merges with stored results for the other cohorts

`03_figures`
- `nh3fig.py`: Figs 2–8 and Extended Data Figs 1–4
- `gen_fig1.py`, `render_fig1.py`: Fig. 1
- `fig_flow.py`, `fig_dag3.py`: participant flow and causal diagram
- `svgfigs.py`: SVG versions of the figures
- `source_data.py`: Source Data workbook

`resources`
- Natural Earth country boundaries for the map in Fig. 1

## Running

Analysis and figure scripts read and write intermediate files (panels, trials, `results_*.csv`) in the working directory, so run them from a single analysis folder with the three code folders on the Python path. The main steps are:

```
python 01_hrs_panel.py            # and the other panel scripts
python build_trials.py
USE_LAG=1 USE_EXTRA=1 python run_main.py hgb 0.025
python repool.py
python nh3fig.py fig2 fig3 fig4 fig5 fig6 fig7n fig8 ed1 ed2 ed3 ed4
```

Random seeds are fixed, so reruns give the same estimates.
