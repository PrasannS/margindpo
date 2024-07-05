export CUDA_VISIBLE_DEVICES=6,7
accelerate launch --config_file=config.yaml --num_processes=2 sft.py