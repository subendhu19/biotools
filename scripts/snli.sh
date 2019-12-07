#!/bin/sh

python run_snli.py --data_dir=/mnt/nfs/work1/mfiterau/brawat/snli/data/snli_1.0 \
--model_type=bert \
--language=english \
--output_dir=/mnt/nfs/work1/mfiterau/brawat/snli/out \
--cache=/mnt/nfs/work1/mfiterau/brawat/snli/cache \
--max_seq_length=200 \
--do_lower_case=True \
--do_train=True \
--do_eval=True