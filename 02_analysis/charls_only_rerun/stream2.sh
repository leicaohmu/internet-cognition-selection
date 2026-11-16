set -x
export PYTHONPATH=/path/to/project/cr OMP_NUM_THREADS=1
cd /path/to/project/cr
python3 newout.py CHARLS > log_newout.txt 2>&1
python3 negctrl.py CHARLS > log_nc.txt 2>&1
python3 negctrl2.py CHARLS > log_nc2.txt 2>&1
python3 subgrp.py CHARLS > log_sub.txt 2>&1
python3 corr_designs.py CHARLS > log_corr.txt 2>&1
python3 behav.py CHARLS > log_beh.txt 2>&1
python3 sustained.py CHARLS > log_sus.txt 2>&1
TRAJ=1 python3 sustained.py CHARLS > log_sustraj.txt 2>&1
python3 dose.py > log_dose.txt 2>&1
python3 dose_lag.py > log_doselag.txt 2>&1
python3 stop_pred.py > log_stoppred.txt 2>&1
python3 vig_pred.py > log_vigpred.txt 2>&1
python3 robust_sel.py > log_robsel.txt 2>&1
python3 anatomy.py > log_anat.txt 2>&1
python3 traj.py > log_traj.txt 2>&1
python3 flows.py > log_flows.txt 2>&1
echo STREAM2 DONE
