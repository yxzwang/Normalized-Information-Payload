CUDA_VISIBLE_DEVICES=2 python run_mlm.py \
    --seed 50 \
    --model_name_or_path ./bert80k-consine/pretraining_experiment-/epoch1000000_step80030 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --fp16 \
    --logging_strategy steps \
    --logging_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 2000 \
    --save_strategy epoch \
    --save_steps 1000 \
    --output_dir ./bertlarge/wikitext-103-raw-v1 \
    --overwrite_output_dir


CUDA_VISIBLE_DEVICES=1 python run_mlm_sparse.py \
    --model_name_or_path ./tbert240k/epoch1000000_step240018 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 2 \
    --fp16 \
    --fp16_full_eval \
    --logging_strategy steps \
    --logging_steps 200 \
    --output_dir ./tbert240kmlm_hyperextension/wikitext-103-raw-v1 \
    --overwrite_output_dir



    --do_eval

    CUDA_VISIBLE_DEVICES=1 python run_mlm_sparse.py \
    --model_name_or_path ./tbert240kmlm/wikitext-103-raw-v1/checkpoint-85500\
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 8 \
    --do_eval \
    --fp16 \
    --fp16_full_eval \
    --logging_strategy steps \
    --logging_steps 200 \
    --output_dir ./eval/tbert240kmlm/wikitext-103-raw-v1 \
    --overwrite_output_dir
