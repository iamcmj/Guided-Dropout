import argparse
import numpy as np
from datasets import load_dataset, concatenate_datasets

import evaluate
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    AdamW,
    get_linear_schedule_with_warmup,

)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


check_min_version("4.22.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--task")
    parser.add_argument("-d", "--output_dir")
    parser.add_argument("-s", "--seed")
    parser.add_argument("-p", "--training_set_pct")
    args = parser.parse_args()
    parser_params = vars(args)

    task_name = parser_params['task']
    max_seq_length = 512
    set_seed(int(parser_params['seed']))
    raw_datasets = load_dataset(
        "glue",
        task_name
    )

    if task_name != 'mnli':
        raw_datasets = concatenate_datasets([raw_datasets['train'], raw_datasets['validation']])
        raw_datasets = raw_datasets.shuffle(seed = int(parser_params['seed']))
        raw_datasets = raw_datasets.train_test_split(test_size=0.2, shuffle=False)
    
    else:
        raw_datasets['train'] = raw_datasets['train'].shuffle(seed = int(parser_params['seed']))
    
    # Labels
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        finetuning_task=task_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-uncased',
        use_fast=True,
        )
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        config=config,
        )
    sentence1_key, sentence2_key = task_to_keys[task_name]
    
    padding = "max_length"
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and task_name is not None
        and not is_regression
    ):
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,  
    )
    
    inds = int((int(parser_params['training_set_pct'])/100) * len(raw_datasets["train"]))
    train_dataset = raw_datasets["train"].select(range(inds))
    if task_name != 'mnli':
        eval_dataset = raw_datasets["test"]
    else:
        eval_dataset = raw_datasets["validation_matched"]
    
    metric = evaluate.load("glue", task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
 
    data_collator = default_data_collator
    training_args = TrainingArguments(output_dir = parser_params['output_dir'], overwrite_output_dir = True)
    training_args.evaluation_strategy = 'epoch'
    training_args.logging_strategy = 'epoch'
    training_args.save_strategy = 'epoch'
    training_args.per_device_train_batch_size = 18
    training_args.seed = int(parser_params['seed'])
    training_args.num_train_epochs = 3
    
    optimizer = AdamW(model.parameters(), lr=2e-05)
    
    #Set model dropout values based on their Fisher Scores
    for idx, m in enumerate(model.modules()):
        if idx == 136:
            m.p = 0.099
        elif idx == 187:
            m.p = 0.09625
        elif idx == 170:
            m.p = 0.0935
        elif idx == 153:
            m.p = 0.09075
        elif idx == 34:
            m.p = 0.088
        elif idx == 119:
            m.p = 0.08525
        elif idx == 204:
            m.p = 0.0825
        elif idx == 51:
            m.p = 0.07975
        elif idx == 16:
            m.p = 0.077
        elif idx == 102:
            m.p = 0.07425
        elif idx == 68:
            m.p = 0.0715
        elif idx == 85:
            m.p = 0.06875
        elif idx == 208:
            m.p = 0.066
        elif idx == 20:
            m.p = 0.06325
        elif idx == 174:
            m.p = 0.0605
        elif idx == 191:
            m.p = 0.05775
        elif idx == 140:
            m.p = 0.055
        elif idx == 38:
            m.p = 0.05225
        elif idx == 123:
            m.p = 0.0495
        elif idx == 157:
            m.p = 0.04675
        elif idx == 55:
            m.p = 0.044
        elif idx == 106:
            m.p = 0.04125
        elif idx == 89:
            m.p = 0.0385
        elif idx == 72:
            m.p = 0.03575
        elif idx == 214:
            m.p = 0.033
        elif idx == 180:
            m.p = 0.03025
        elif idx == 129:
            m.p = 0.0275
        elif idx == 112:
            m.p = 0.02475
        elif idx == 61:
            m.p = 0.022
        elif idx == 27:
            m.p = 0.01925
        elif idx == 197:
            m.p = 0.0165
        elif idx == 163:
            m.p = 0.01375
        elif idx == 146:
            m.p = 0.011
        elif idx == 44:
            m.p = 0.00825
        elif idx == 95:
            m.p = 0.0055
        elif idx == 78:
            m.p = 0.00275
        elif idx == 7:
            m.p = 0.0
            
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers = (optimizer, None)
    )

    # Training
    training_args_resume_from_checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=training_args_resume_from_checkpoint)
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    tasks = [task_name]
    eval_datasets = [eval_dataset]

    if task_name == "mnli":
        tasks.append("mnli-mm")
        eval_datasets.append(raw_datasets["validation_mismatched"])
        combined = {}
    
    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        if task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        if task is not None and "mnli" in task:
            combined.update(metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

if __name__ == "__main__":
    main()