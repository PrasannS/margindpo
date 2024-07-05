export TRANSFORMERS_CACHE=cache/

deepspeed --include localhost:4,5,6,7 openrlhf_sft.py \
   --save_path ./cache/pythia-ufsft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps 500 \
   --train_batch_size 256 \
   --micro_train_batch_size 4 \
   --pretrain "EleutherAI/pythia-2.8b" \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset cache/ufsft \
   --apply_chat_template \
   --input_key chosen \
   --output_key chosen \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb True \
   --wandb_run_name sftvanilla