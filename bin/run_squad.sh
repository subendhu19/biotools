#!/usr/bin/env bash

CHECKPOINT=$1
TYPE=$2
BASE=$3
OUTPUT_SUFFIX=$4
#
#SBATCH --job-name=squad
#SBATCH --output=squad_${OUTPUT_SUFFIX}.txt
#SBATCH --time=0-10:00

python scripts/run_squad.py \
  --model_type ${TYPE} \
  --model_name_or_path ${CHECKPOINT} \
  --config_name ${BASE} \
  --tokenizer_name ${BASE} \
  --do_train \
  --do_eval \
  --train_file data/squad/train-v1.1.json \
  --predict_file data/squad/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --logging_steps=2500 \
  --save_steps=2500 \
  --output_dir checkpoints/squad_${OUTPUT_SUFFIX}/

