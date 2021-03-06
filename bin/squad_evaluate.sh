#!/bin/bash
#SBATCH -n 5
#SBATCH -N 1
#SBATCH --mem=10000

#SBATCH -o /mnt/nfs/work1/hongyu/abhyuday/logs/evl/exp_%A_%a.out
#SBATCH -e /mnt/nfs/work1/hongyu/abhyuday/logs/evl/exp_%A_%a.err

module load cuda92
eval "$('/home/abhyuday/local/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
source activate cuda92

echo "Starting the execution of task $SLURM_JOBID"
echo $CUDA_VISIBLE_DEVICES



INPUTFILE=/mnt/nfs/scratch1/abhyuday/models.csv
TRAINF=/home/abhyuday/data/squad/train-v2.0.json
TESTF=/home/abhyuday/data/squad/dev-v2.0.json
OUTPUTDIR=/home/abhyuday/scratch/cl_evl/chkpnts


LINE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $INPUTFILE)
IFS=',' read CHECKPOINT MODELTYPE MODELNAME EXTRA INSTANCENAME <<<"$LINE"
#echo $LINE
echo " MODELPATH: ${CHECKPOINT}"
echo $MODELTYPE
echo $MODELNAME
echo $EXTRA
echo $INSTANCENAME

cd /home/abhyuday/src/biotools

python scripts/run_squad.py \
  --model_type $MODELTYPE \
  --model_name_or_path ${CHECKPOINT} \
  --config_name $MODELNAME \
  --tokenizer_name $MODELNAME \
  --do_train \
  --do_eval \
  --train_file $TRAINF \
  --predict_file $TRAINF \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps=10000 \
  --output_dir $OUTPUTDIR/squad_${INSTANCENAME}/ \
  --version_2_with_negative \
  --overwrite_output_dir \
  --overwrite_cache
  ${EXTRA}

echo $V

echo "Execution ended for ${SLURM_JOBID} : ${SLURM_ARRAY_TASK_ID}"


