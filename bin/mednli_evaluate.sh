#!/bin/bash
#SBATCH -n 5
#SBATCH -N 1
#SBATCH --mem=40000

#SBATCH -o /mnt/nfs/scratch1/srongali/mednli_logs/exp_%A_%a.out
#SBATCH -e /mnt/nfs/scratch1/srongali/mednli_logs/exp_%A_%a.err

echo "Starting the execution of task $SLURM_JOBID"
echo $CUDA_VISIBLE_DEVICES


INPUTFILE=/mnt/nfs/scratch1/abhyuday/modelsv2.csv
DATADIR=/mnt/nfs/work1/mfiterau/brawat/snli/data/mnli/mednli/mednli_data
OUTPUTDIR=//mnt/nfs/scratch1/srongali/biotools/checkpoints/mednli


LINE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $INPUTFILE)
IFS=',' read CHECKPOINT MODELTYPE MODELNAME EXTRA INSTANCENAME <<<"$LINE"
#echo $LINE
echo " MODELPATH: ${CHECKPOINT}"
echo $MODELTYPE
echo $MODELNAME
echo $EXTRA
echo $INSTANCENAME

cd /home/srongali/Projects/bioembeddings/biotools

python scripts/run_snli.py --data_dir=$DATADIR \
--model_type=$MODELTYPE \
--model_name_or_path=${CHECKPOINT} \
--config_name $MODELNAME \
--tokenizer_name $MODELNAME \
--language=english \
--output_dir=$OUTPUTDIR/snli_${INSTANCENAME}/ \
--max_seq_length=200 \
--do_train \
--do_eval \
--save_steps=15000 \
--num_train_epochs=3 \
--overwrite_output_dir \
--overwrite_cache \
--per_gpu_train_batch_size=20 \
--per_gpu_eval_batch_size=20 ${EXTRA}

echo $V

echo "Execution ended for ${SLURM_JOBID} : ${SLURM_ARRAY_TASK_ID}"


