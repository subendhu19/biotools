#!/bin/bash
#SBATCH -n 5
#SBATCH -N 1
#SBATCH --mem=30000

#SBATCH -o /mnt/nfs/scratch1/srongali/emrqa_logs/exp_%A_%a.out
#SBATCH -e /mnt/nfs/scratch1/srongali/emrqa_logs/exp_%A_%a.err

echo "Starting the execution of task $SLURM_JOBID"
echo $CUDA_VISIBLE_DEVICES



INPUTFILE=/mnt/nfs/scratch1/abhyuday/modelsv2.csv
TRAINF=/mnt/nfs/work1/mfiterau/brawat/emrqa_data_biotool/data_squad_biotools/train.json
TESTF=//mnt/nfs/work1/mfiterau/brawat/emrqa_data_biotool/data_squad_biotools/test.json
OUTPUTDIR=//mnt/nfs/scratch1/srongali/biotools/checkpoints/emrqa


LINE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $INPUTFILE)
IFS=',' read CHECKPOINT MODELTYPE MODELNAME EXTRA INSTANCENAME <<<"$LINE"
#echo $LINE
echo " MODELPATH: ${CHECKPOINT}"
echo $MODELTYPE
echo $MODELNAME
echo $EXTRA
echo $INSTANCENAME

cd /home/srongali/Projects/bioembeddings/biotools

python scripts/run_squad.py \
  --model_type $MODELTYPE \
  --model_name_or_path ${CHECKPOINT} \
  --config_name $MODELNAME \
  --tokenizer_name $MODELNAME \
  --do_train \
  --do_eval \
  --train_file $TRAINF \
  --predict_file $TESTF \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps=200000 \
  --output_dir $OUTPUTDIR/squad_${INSTANCENAME}/ \
  --overwrite_output_dir \
  --overwrite_cache ${EXTRA}

echo $V

echo "Execution ended for ${SLURM_JOBID} : ${SLURM_ARRAY_TASK_ID}"


