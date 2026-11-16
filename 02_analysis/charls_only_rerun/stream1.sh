set -x
export PYTHONPATH=/path/to/project/cr OMP_NUM_THREADS=1
cd /path/to/project/cr
USE_LAG=1 USE_EXTRA=1 python3 run_main.py hgb 0.025 > log_rm4.txt 2>&1
USE_LAG=1 USE_EXTRA=1 python3 placebo.py > log_pl.txt 2>&1
USE_LAG=1 USE_EXTRA=1 USE_T1=1 python3 placebo.py > log_pl_t1.txt 2>&1
python3 placebo2.py > log_pl2.txt 2>&1
python3 within.py > log_wi.txt 2>&1
python3 within2.py > log_wi2.txt 2>&1
python3 within3.py > log_wi3.txt 2>&1
python3 within4.py > log_wi4.txt 2>&1
python3 pretrend.py > log_pre.txt 2>&1
python3 round3.py > log_r3.txt 2>&1
python3 diff_tte_placebo.py > log_diff.txt 2>&1
python3 sens_overlap.py > log_sens.txt 2>&1
python3 attrition.py > log_attr.txt 2>&1
python3 run_t1main.py > log_t1m.txt 2>&1
echo STREAM1A DONE
python3 run_death.py base > log_d1.txt 2>&1
USE_LAG=1 python3 run_death.py lag > log_d2.txt 2>&1
USE_LAG=1 USE_EXTRA=1 python3 run_death.py primary > log_d3.txt 2>&1
USE_LAG=1 USE_T1=1 USE_EXTRA=1 python3 run_death.py conservative > log_d4.txt 2>&1
echo STREAM1B DONE
export MAIN_Y=Y_recall,Y_iadl,Y_dep,Y_death,Y_cancer SKIPBAL=1
python3 run_main.py hgb 0.025 > log_rm2.txt 2>&1
USE_LAG=1 python3 run_main.py hgb 0.025 > log_rm3.txt 2>&1
USE_LAG=1 USE_T1=1 USE_EXTRA=1 python3 run_main.py hgb 0.025 > log_rm6.txt 2>&1
export MAIN_Y=Y_recall
python3 run_main.py hgb > log_rm1.txt 2>&1
USE_LAG=1 USE_T1=1 python3 run_main.py hgb 0.025 > log_rm5.txt 2>&1
echo STREAM1C DONE
