#!/usr/bin/env bash

CHECKPOINT=$1
TYPE=$2
BASE=$3
OUTPUT_SUFFIX=$4
#
#SBATCH --job-name=xnli
#SBATCH --output=xnli_${OUTPUT_SUFFIX}.txt
#SBATCH --time=0-10:00

python scripts/run_xnli.py \
  --model_type ${TYPE} \
  --model_name_or_path ${CHECKPOINT} \
  --config_name ${BASE} \
  --tokenizer_name ${BASE} \
  --language de \
  --train_language en \
  --do_train \
  --do_eval \
  --data_dir data/xnli \
  --per_gpu_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --output_dir checkpoints/xnli_${OUTPUT_SUFFIX} \
  --logging_steps=2500 \
  --save_steps 2500