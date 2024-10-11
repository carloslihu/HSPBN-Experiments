#!/bin/bash
# train + test
python abalone.py
echo "abalone.py done"

python adult.py
echo "adult.py done"

python australian_statlog.py
echo "australian_statlog.py done"

python cover_type.py
echo "cover_type.py done"

python credit_approval.py
echo "credit_approval.py done"

python german_statlog.py
echo "german_statlog.py done"

python kdd.py
echo "kdd.py done"

python liver_disorders.py
echo "liver_disorders.py done"

python thyroid_hypothyroid.py
echo "thyroid_hypothyroid.py done"

python thyroid_sick.py
echo "thyroid_sick.py done"

# plot
python plot_results.py
echo "plot_results.py done"