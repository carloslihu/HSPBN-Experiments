#!/bin/bash
# train
python train_hc_clg.py
python train_hc_hspbn.py 
# python train_hc_hspbn_hckde.py 

# test
python test_hc_clg.py
python test_hc_hspbn.py 
# python test_hc_hspbn_hckde.py
python test_hc_times.py 