# Normalized Information Payload
 Codes for paper: [What Dense Graph Do You Need for Self-attention?](https://arxiv.org/abs/2205.14014)。
## Requirements
The whole environment is in whole_environment.txt. For some main packages, we use 
`deepspeed==0.5.4`
`torch==1.9.1+cu111`
`transformers==4.10.0`
`datasets==1.14.0`.
## Usage:
There are two parts of experiments, the [Long-Range-Arena](https://github.com/google-research/long-range-arena)(LRA) and BERT pretraining.
### LRA
For LRA tasks, we construct our codes based on Nystromformer([paper](https://arxiv.org/abs/2102.03902), [Github](https://github.com/mlpen/Nystromformer)), run task by 

`python run_tasks.py --model attn_type --task taskname --seed seed`

where we support attn_type in ["hypercube", "bigbird", "longformer", "global", "local", "random", "local+random"], taskname in ["listops","text","retrieval","image","pathfinder32-curv_contour_length_14"].

Datasets should be put in LRA/datasets/ and can be downloaded from [here]()(download link coming soon). You can also use datasets from [Nystromformer LRA.](https://github.com/mlpen/Nystromformer/tree/main/LRA)

### BERT Pretraining and finetuning
Codes are in Pretraining_finetuning. For BERT pretraining, we adopt the method from academic-budget-bert([paper](https://arxiv.org/abs/2104.07705), [Github](https://github.com/IntelLabs/academic-budget-bert)). Full guides can be found there.  
For finetuning, run 
`python run_glue_sparse.py \
  --model_name_or_path modelpath\
  --task_name mrpc \
  --max_seq_length 128 \
  --seed 42 \
  --output_dir outputdir \
  --overwrite_output_dir \
  --do_train \
  --fp16 \
  --fp16_full_eval \
  --do_eval \
  --eval_steps 1000 \
  --do_predict \
  --save_strategy steps \
  --save_steps 1000 \
  --metric_for_best_model accuracy \
  --evaluation_strategy steps \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --num_train_epochs 5 \
  --lr_scheduler_type polynomial \
  --warmup_steps 0`
  
or

`python run_mlm_sparse.py \
    --seed 42
    --model_name_or_path modelpath\
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
    --output_dir output_dir \
    --overwrite_output_dir`
.
