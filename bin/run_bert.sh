#!/usr/bin/env bash
#
#SBATCH --job-name=bio_finetune
#SBATCH --output=bio_bert_dis.txt
#SBATCH --time=7-00:00

python scripts/run_lm_finetuning.py \
    --output_dir=checkpoints/bio-bert-dis \
    --model_type=bert \
    --model_name_or_path=bert-base-cased \
    --do_train \
    --train_data_file=data/train1.txt \
    --do_eval \
    --eval_data_file=data/valid.txt \
    --evaluate_during_training \
    --logging_steps=25000 \
    --save_steps=25000 \
    --mlm \
    --num_train_epochs=10 \
    --save_total_limit=10 \
    --per_gpu_train_batch_size=3 \
    --per_gpu_eval_batch_size=6

