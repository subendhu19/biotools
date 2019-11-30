#!/usr/bin/env bash
#
#SBATCH --job-name=bio_finetune
#SBATCH --output=bio_distilbert.txt
#SBATCH --time=7-00:00

python scripts/run_lm_finetuning.py \
    --output_dir=checkpoints/bio-distilbert \
    --model_type=distilbert \
    --model_name_or_path=distilbert-base-uncased \
    --do_train \
    --do_lower_case \
    --train_data_file=data/train.txt \
    --do_eval \
    --evaluate_during_training \
    --logging_steps=25000 \
    --save_steps=25000 \
    --eval_data_file=data/valid.txt \
    --mlm \
    --num_train_epochs=10 \
    --save_total_limit=10 \
    --per_gpu_train_batch_size=5 \
    --per_gpu_eval_batch_size=10
