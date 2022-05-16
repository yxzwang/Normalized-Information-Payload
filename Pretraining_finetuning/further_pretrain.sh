export WANDB_API_KEY=e4bc15547627f21afb893e4a9de53a0478dbeb6d

#deepspeed run_pretraining_sparse.py \
deepspeed run_pretraining.py \
    --model_type 'bert-mlm' \
    --tokenizer_name './bert-large-uncased-vocab.txt' \
    --hidden_act 'gelu' \
    --hidden_size '1024' \
    --num_hidden_layers '24' \
    --num_attention_heads '16' \
    --intermediate_size '4096' \
    --hidden_dropout_prob '0.1' \
    --attention_probs_dropout_prob '0.1' \
    --encoder_ln_mode 'pre-ln' \
    --lr '5e-5' \
    --train_batch_size '4032' \
    --train_micro_batch_size_per_gpu '14' \
    --lr_schedule 'step' \
    --max_steps '8000' \
    --curve 'cosine' \
    --warmup_proportion '0.02' \
    --gradient_clipping '0.0' \
    --optimizer_type 'adamw' \
    --weight_decay '0.01' \
    --adam_beta1 '0.9' \
    --adam_beta2 '0.98' \
    --adam_eps '1e-6' \
    --dataset_path './bert_data_bin_512_dup40' \
    --output_dir './bert80k+8k-128->512' \
    --print_steps '1000' \
    --num_epochs_between_checkpoints '10000' \
    --job_name 'pretraining_experiment' \
    --project_name 'budget-bert-pretraining' \
    --validation_epochs '80' \
    --validation_epochs_begin '1' \
    --validation_epochs_end '1' \
    --validation_begin_proportion '0.05' \
    --validation_end_proportion '0.01' \
    --validation_micro_batch '8' \
    --deepspeed \
    --data_loader_type 'dist' \
    --do_validation \
    --seed '42' \
    --fp16 \
    --early_exit_time_marker '1200000' --total_training_time '1200000'