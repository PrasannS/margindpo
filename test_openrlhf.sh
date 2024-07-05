deepspeed --include localhost:4,5,6,7 openrlhf_dpo.py \
   --save_path ./cache/pythia-ufnormal \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps 500 \
   --train_batch_size 256 \
   --micro_train_batch_size 4 \
   --pretrain yaswanthchittepu/pythia2.8b-ultrafeedback-binarized-sft \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --beta 0.1 \
   --dataset cache/ufprefs \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb True \
   --wandb_run_name dpovanilla
   
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # --ipo [for IPO]
     # --label_smoothing 0.1 [for cDPO]
     # --ref_offload 

# Customization of chat_template is also supported.
# --apply_chat_template 
# --input_key {JSON Key}
# --tokenizer_chat_template {HF Chat Template}