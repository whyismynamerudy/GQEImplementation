#!/bin/bash
#SBATCH --partition=ashton
#SBACTH --qos=ashton
#SBATCH --job-name=mongarud-manual
#SBATCH --output=manual/log_%j.log      # Standard output and error log
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

mkdir -p manual

conda activate gqeimplementation
python3 main.py --do_train --do_valid --do_test --data_path "NELL-betae" --num_epochs 40 --batch_size 200 --test_batch_size 75 --learning_rate 0.0001