#!/bin/sh

python run_snli.py --data_dir=/mnt/nfs/work1/mfiterau/brawat/snli/data/snli_1.0 \
--model_type=bert \
--model_name_or_path=bert-base-uncased \
--language=english \
--output_dir=/mnt/nfs/work1/mfiterau/brawat/snli/ckpts/check \
--cache=/mnt/nfs/work1/mfiterau/brawat/snli/cache \
--max_seq_length=200 \
--do_lower_case \
--do_train \
--do_eval \
--save_steps=15000 \
--num_train_epochs=3 \
--warmup_steps=9000 \
--overwrite_output_dir \
--per_gpu_train_batch_size=20 \
--per_gpu_eval_batch_size=20 \
--max_steps=30