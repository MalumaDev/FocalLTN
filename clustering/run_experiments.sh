#!/bin/bash
for j in 1. 0.75 0.5 0.25 0.1
do
  for i in 1 2 3 4 5
  do
      python script-stableproduct-ltn.py --csv-path "results/stable_product_p2_$i.csv" --p 2 --seed $i --imbalance $j
      python script-stableproduct-ltn.py --csv-path "results/stable_product_p6_$i.csv" --p 6 --seed $i --imbalance $j
  #    python prod_rl.py --csv-path "results/1500/prod_rl_$i.csv" --n-examples-train 1500 --seed $i
      python script-focal-ltn.py --csv-path "results/logltn_default_$i.csv" --seed $i --imbalance $j
      python script-focal-ltn.py --csv-path "results/logltn_focal_$i.csv" --seed $i --use_focal --imbalance $j
  #    python log_ltn_max.py --csv-path "results/1500/logltn_max_$i.csv" --n-examples-train 1500 --seed $i
  #    python log_ltn_lseup.py --csv-path "results/1500/logltn_lseup_$i.csv" --n-examples-train 1500 --seed $i
  #    python log_ltn_sum.py --csv-path "results/1500/logltn_sum_$i.csv" --n-examples-train 1500 --seed $i
  done
done

#for i in 1 2 3 4 5
#do
#    python stable_product.py --csv-path "results/15000/stable_product_p2_$i.csv" --n-examples-train 15000 --p 2 --seed $i
#    python stable_product.py --csv-path "results/15000/stable_product_p6_$i.csv" --n-examples-train 15000 --p 6 --seed $i
##    python prod_rl.py --csv-path "results/15000/prod_rl_$i.csv" --n-examples-train 15000 --seed $i
#    python log_ltn.py --csv-path "results/15000/logltn_default_$i.csv" --n-examples-train 15000 --seed $i
#    python log_ltn.py --csv-path "results/15000/logltn_default_$i.csv" --n-examples-train 15000 --seed $i --use_focal
##    python log_ltn_max.py --csv-path "results/15000/logltn_max_$i.csv" --n-examples-train 15000 --seed $i
##    python log_ltn_lseup.py --csv-path "results/15000/logltn_lseup_$i.csv" --n-examples-train 15000 --seed $i
##    python log_ltn_sum.py --csv-path "results/15000/logltn_sum_$i.csv" --n-examples-train 15000 --seed $i
#done