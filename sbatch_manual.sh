#!/bin/bash
#SBATCH --partition=ashton
#SBACTH --qos=ashton
#SBATCH --job-name=mongarud-manual
#SBATCH --output=manual/log_1p2p3p_with_llm_and_skipping_%j.log      # Standard output and error log
#SBATCH --time=12:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:1

mkdir -p manual

conda activate gqeimplementation
python3 main.py --load_model --model_path runs/2024-06-14_01-31-43/2024-06-14_01-31-43_model.pth --data_path NELL-betae/ --tasks 1p.2p.3p --use_llm