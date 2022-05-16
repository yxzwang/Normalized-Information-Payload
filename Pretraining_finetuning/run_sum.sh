CUDA_VISIBLE_DEVICES=6 python run_summarization_sparse.py \
  --model_name_or_path facebook/bart-large \
  --dataset_name ccdv/pubmed-summarization \
  --fp16 \
  --fp16_full_eval \
  --max_source_length 1024 \
  --num_train_epochs 16 \
  --learning_rate 5e-5 \
  --do_train \
  --do_predict \
  --pad_to_max_length \
  --predict_with_generate \
  --save_strategy epoch \
  --logging_strategy steps\
  --logging_steps 500 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 4 \
  --overwrite_output_dir \
  --output_dir ./outputs/hyperBART--ccdv/pubmed-summarization \


  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 100 


  CUDA_VISIBLE_DEVICES=0 python run_summarization_sparse.py \
  --model_name_or_path ./outputs/hyperBART--ccdv/pubmed-summarization/checkpoint-359772 \
  --dataset_name ccdv/pubmed-summarization \
  --fp16 \
  --fp16_full_eval \
  --max_source_length 1024 \
  --num_train_epochs 16 \
  --learning_rate 5e-5 \
  --do_eval \
  --pad_to_max_length \
  --predict_with_generate \
  --save_strategy epoch \
  --logging_strategy steps\
  --logging_steps 500 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 4 \
  --overwrite_output_dir \
  --output_dir ./outputs/hyperBART--ccdv/pubmed-summarization/test \
###############################################################################################################

CUDA_VISIBLE_DEVICES=4 python run_summarization_sparse.py \
  --model_name_or_path facebook/bart-large \
  --dataset_name ccdv/arxiv-summarization \
  --fp16 \
  --fp16_full_eval \
  --max_source_length 1024 \
  --num_train_epochs 16 \
  --learning_rate 5e-5 \
  --do_train \
  --do_predict \
  --pad_to_max_length \
  --predict_with_generate \
  --save_strategy epoch \
  --logging_strategy steps\
  --logging_steps 500 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 4 \
  --overwrite_output_dir \
  --output_dir ./outputs/hyperBART--ccdv/arxiv-summarization \


CUDA_VISIBLE_DEVICES=5 python run_summarization_sparse.py \
  --model_name_or_path facebook/bart-large \
  --dataset_name cnn_dailymail \
  --dataset_config_name 3.0.0 \
  --fp16 \
  --fp16_full_eval \
  --max_source_length 1024 \
  --num_train_epochs 16 \
  --learning_rate 5e-5 \
  --do_train \
  --do_predict \
  --pad_to_max_length \
  --predict_with_generate \
  --save_strategy epoch \
  --logging_strategy steps\
  --logging_steps 500 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 1 \
  --overwrite_output_dir \
  --output_dir ./outputs/hyperBART--ccdv/arxiv-summarization \
  --output_dir ./outputs/hyperBART--cnn_dailymail \
  

  
