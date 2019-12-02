#!/usr/bin/env bash

sbatch -p 1080ti-long --mem 380000 --gres=gpu:8 --cpus-per-task=8 bin/run_distilbert.sh
