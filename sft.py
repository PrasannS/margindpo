from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
import sys, os, datetime

if __name__ == "__main__":

    ######################### ARGS ###########################
    model_name = "EleutherAI/pythia-2.8b"
    dataset_name = "zhengr/ultrafeedback_binarized"
    cache_dir = "cache/"
    out_dir = f"cache/"
    batch_size = 8
    gradient_accumulation_steps = 8
    gradient_checkpoint = True
    lr = 5e-7
    scheduler = 'cosine'
    num_epochs = 4
    max_seq_length = 2048
    ##########################################################

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    dataset = dataset.filter(
        lambda x: len(x['messages'][-1]['content']) > 0
    )

    train_data = dataset['train_sft']
    test_data = dataset['test_sft']
    
    train_data = train_data.map(lambda batch: {'completion': [msg[-1]['content'] for msg in batch['messages']]}, batched=True, 
                    remove_columns=['prompt_id','chosen','rejected','messages','score_chosen','score_rejected'])
    test_data = test_data.map(lambda batch: {'completion': [msg[-1]['content'] for msg in batch['messages']]}, batched=True, 
                    remove_columns=['prompt_id','chosen','rejected','messages','score_chosen','score_rejected'])

    train_data = train_data.select(range(256))
    test_data = test_data.select(range(256))

    # Increasing timeout is critical when working with big data/models
    # Otherwise errors out due to socket time out. Default timeout is 3600 seconds
    # torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=36000))

    tokenizer = AutoTokenizer.from_pretrained(model_name,)
    tokenizer.pad_token = tokenizer.eos_token # Code errors out without this

    train_data = train_data.filter(
        lambda x: len(tokenizer(f"Prompt: {x['prompt']}\n### Answer: {x['completion']}")["input_ids"]) <= max_seq_length
    )
    test_data = test_data.filter(
        lambda x: len(tokenizer(f"Prompt: {x['prompt']}\n### Answer: {x['completion']}")["input_ids"]) <= max_seq_length
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                cache_dir=cache_dir, 
                                                # device_map='auto',
                                                device_map={"": Accelerator().local_process_index},
                                                attn_implementation="flash_attention_2",
                                                torch_dtype=torch.bfloat16
                                            )
    model.config.pad_token_id = model.config.eos_token_id # Code errors out without this
    # Set this back to True at Inference time
    model.config.use_cache = False # To supress warnings

    # THIS IS DIFFERENT FROM HOW WE PROCESS NORMAL SFT WITH THE FORMAT {'prompt': xxxxxx, 'response': yyyyyy}
    # THIS IS BECAUSE OF DataCollatorForCompletionOnlyLM
    # DataCollatorForCompletionOnlyLM CALLS A MAP METHOD THAT BATCHES THE DATA AND PROCESSES THE DATA. HENCE WE NEED THE LOOP
    def formatting_prompts_func(data):
        output_texts = []
        for i in range(len(data['prompt'])):
            text = f"Prompt: {data['prompt'][i]}\n### Answer: {data['completion'][i]}"
            output_texts.append(text)
        return output_texts

    response_template = "\n### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer,)

    if(gradient_checkpoint):
        # Needed when we use gradient checkpointing
        # Otherwise complains that no gradients to compute 
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    training_args = TrainingArguments(
        output_dir=out_dir,
        report_to="none",  # this tells the Trainer to log the metrics to W&B
        run_name = f"pythia-2.8-ultrafeedback-sft", # Sets the wandb run name
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        bf16=True, # Need A100s
        learning_rate=lr,
        lr_scheduler_type=scheduler,
        warmup_ratio = 0.1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant":False}, # Important, otherwise errors out if using accelerate
        evaluation_strategy="steps",
        eval_steps=50,
        num_train_epochs=num_epochs,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps = 50,
        save_total_limit = 5,
        load_best_model_at_end = True,
        max_grad_norm=10.,
        remove_unused_columns=True, # IMPORTANT: This needs to be set to True for SFT
        # Otherwise we get the following error:
        # ValueError: Unable to create tensor, you should probably activate truncation and/or
        # padding with 'padding=True' 'truncation=True' to have batched tensors with the same
        # length. Perhaps your features (`input_ids` in this case) have excessive nesting (inputs
        # type `list` where type `int` is expected).
        # metric_for_best_model='accuracy' # Saves best model based on accuracy instead of the loss
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=max_seq_length,
        packing=False,
    )

    trainer.train()

    # This is needed to load the full state dict, when using FSDP
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    full_save_dir = os.path.join(out_dir, 'full_save')
    os.makedirs(full_save_dir, exist_ok=True)
    trainer.save_model(full_save_dir)

