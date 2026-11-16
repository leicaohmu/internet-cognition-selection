while pgrep -f stream2.sh > /dev/null; do sleep 30; done
export PYTHONPATH=/path/to/project/cr OMP_NUM_THREADS=1
cd /path/to/project/cr
python3 ladder.py > log_ladder.txt 2>&1
python3 hte.py > log_hte.txt 2>&1
cd /path/to/project/an2 && cp results_gradient.csv results_gradient_old.csv && cp results_gradient_metareg.csv results_gradient_metareg_old.csv && python3 gradient.py > log_gradient_w6.txt 2>&1
echo STREAM3 DONE
