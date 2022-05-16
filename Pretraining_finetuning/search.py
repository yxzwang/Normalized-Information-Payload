from datasets import load_dataset, load_metric
from transformers import  AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, default_data_collator, DataCollatorWithPadding

from pretraining.modeling import BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
dataset = load_dataset('glue', 'mrpc')
metric = load_metric('glue', 'mrpc')

encoded_dataset = dataset.map(
    lambda b:tokenizer(b['sentence1'], b['sentence2'], padding="max_length", max_length=128, truncation=True), 
    batched=True
)

def model_init():
    return BertForSequenceClassification.from_pretrained("bert80k/pretraining_experiment-/epoch1000000_step80030/")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    m = metric.compute(predictions=predictions, references=labels)
    return m

training_args = TrainingArguments(
    evaluation_strategy="steps", eval_steps=200, disable_tqdm=True, output_dir='/tmp/out',
    save_steps=200, save_total_limit=1, # need this to make optuna working
    max_grad_norm=1.0,
    fp16=True,
    load_best_model_at_end=True, metric_for_best_model="accuracy", # need this to perform early stopping
    report_to="none"
)

trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    model_init=model_init,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience = 3)],
    data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) if training_args.fp16 else default_data_collator
)


def hp_space0(trial):
    return {
        # "learning_rate": trial.suggest_float("learning_rate", low=1e-5, higt=8e-5, log=True),
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 3e-5, 5e-5, 8e-5]),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.1]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [3, 5, 10]),
        "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.0, 0.06]),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["polynomial", "linear"]),
    }

def hp_space1(trial):
    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [5e-5, 8e-5]),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [32]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.1]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [3, 5]),
        "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.0, 0.06]),
    }


TASK_NAME_TO_HYPER_PARMS_SPACE_FN = {
    "rte": hp_space0,
    "sst2": hp_space0,
    "mrpc": hp_space0,
    "cola": hp_space0,
    "sts": hp_space0,
    "mnli": hp_space1,
    "qqp": hp_space1,
    "qnli": hp_space1,
}

trainer.hyperparameter_search(
    direction="maximize", 
    backend="optuna", 
    n_trials=30, # number of trials
    hp_space=TASK_NAME_TO_HYPER_PARMS_SPACE_FN["mrpc"]
)
