#!/bin/sh

python run_snli.py --data_dir=/mnt/nfs/work1/mfiterau/brawat/snli/data/snli_1.0 \
--model_=bert-base-uncased \
--model_name_or_path=bert \
--language=english \
--output_dir=/mnt/nfs/work1/mfiterau/brawat/snli/out \
--cache=/mnt/nfs/work1/mfiterau/brawat/snli/cache \
--max_seq_length=200 \
--do_lower_case \
--do_train \
--do_eval