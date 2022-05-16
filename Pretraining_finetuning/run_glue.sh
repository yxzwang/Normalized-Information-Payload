
  CUDA_VISIBLE_DEVICES=4 python run_glue_predict.py \
  --model_name_or_path ./bert80k-consine/pretraining_experiment-/epoch1000000_step80030 \
  --task_name sst2  \
  --max_seq_length 128 \
  --seed 42 \
  --load_best_model_at_end \
  --output_dir ./bert80kcosine/batch32/sst2/5e-5/testspeed \
  --overwrite_output_dir \
  --do_train --do_eval 
  --save_strategy none \

  --evaluation_strategy steps \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --eval_steps 1000 \
  --max_grad_norm 1.0 \
  --num_train_epochs 5 \
  --lr_scheduler_type polynomial \
  --warmup_steps 50 

  CUDA_VISIBLE_DEVICES=5 python run_glue_predict.py \
  --model_name_or_path ./bert80k-consine/pretraining_experiment-/epoch1000000_step80030 \
  --task_name sst2  \
  --max_seq_length 128 \
  --seed 42 \
  --load_best_model_at_end \
  --output_dir ./bert80kcosine/batch32/sst2/2e-5 \
  --overwrite_output_dir \
  --do_train --do_eval --do_predict\
  --save_strategy steps \
    --save_steps 1000 \
  --metric_for_best_model accuracy \
  --evaluation_strategy steps \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --eval_steps 1000 --evaluation_strategy steps \
  --max_grad_norm 1.0 \
  --num_train_epochs 5 \
  --lr_scheduler_type polynomial \
  --warmup_steps 50 

    CUDA_VISIBLE_DEVICES=6 python run_glue_predict.py \
  --model_name_or_path ./bert80k-consine/pretraining_experiment-/epoch1000000_step80030 \
  --task_name sst2  \
  --max_seq_length 128 \
  --seed 42 \
  --load_best_model_at_end \
  --output_dir ./bert80kcosine/batch16/sst2/5e-5 \
  --overwrite_output_dir \
    --save_steps 1000 \
  --do_train --do_eval --do_predict\
  --save_strategy steps \
   --metric_for_best_model accuracy \
  --evaluation_strategy steps \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --eval_steps 1000 --evaluation_strategy steps \
  --max_grad_norm 1.0 \
  --num_train_epochs 5 \
  --lr_scheduler_type polynomial \
  --warmup_steps 50 

  CUDA_VISIBLE_DEVICES=3 python run_glue.py \
  --model_name_or_path ./bert80k-consine/pretraining_experiment-/epoch1000000_step80030 \
  --task_name sst2  \
  --max_seq_length 128 \
  --seed 42 \
  --load_best_model_at_end \
  --output_dir ./bert80kcosine/batch16/sst2/2e-5 \
  --overwrite_output_dir \
  --do_train --do_eval --do_predict\
  --save_strategy steps \
  --metric_for_best_model accuracy \
  --evaluation_strategy steps \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --eval_steps 1000 --evaluation_strategy steps \
  --max_grad_norm 1.0 \
  --num_train_epochs 5 \
  --lr_scheduler_type polynomial \
  --warmup_steps 50 


 