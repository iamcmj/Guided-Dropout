import torch
import argparse
import deepspeed
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets

import evaluate
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    PretrainedConfig,
    DataCollatorForLanguageModeling,
    set_seed,
    AdamW,

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

class GaussianDropout(torch.nn.Module):
    def __init__(self, p=0.1):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([p/(1-p)])
        
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size(), device=x.device) * self.alpha.to(x.device) + 1
            return x * epsilon
        else:
            return x

def replace_layer_for_gaussian_dropout(module):
    if isinstance(module, torch.nn.Dropout):
        return GaussianDropout(p=0.1)
    else:
        return module

def recursive_setattr(obj, attr, value):
    attr = attr.split('.', 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)        

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--task")
    parser.add_argument("-d", "--output_dir")
    parser.add_argument("-s", "--seed")
    parser.add_argument("-p", "--training_set_pct")
    # local rank
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank passed by deepspeed")
    
    args = parser.parse_args()
    parser_params = vars(args)

    task_name = parser_params['task']
    max_seq_length = 512
    set_seed(int(parser_params['seed']))
    
    # load dataset
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
    # check if it's regression or classification
    config = AutoConfig.from_pretrained(
        'meta-llama/Llama-3.2-3B',
        num_labels=num_labels,
        finetuning_task=task_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-3.2-3B',
        use_fast=False,
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-3B',
        config=config,
        torch_dtype=torch.bfloat16
        )
    
    # deepspeed setting
    batch_size = 64
    num_gpus = torch.cuda.device_count()
    grad_accumulation = 4
    micro_batch_size = batch_size // (num_gpus * grad_accumulation)
    assert batch_size % (num_gpus * grad_accumulation) == 0, "(batch_size / (num_gpus * grad_accumulation)) is not integer"
    
    ds_config = {
    "train_batch_size": batch_size,
    "fp16": {
        "enabled": False,
    },
    "bf16": {
        "enabled": True
    },
    "gradient_accumulation_steps": grad_accumulation,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
        },
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 3e-05,
            "betas": [
                0.9,
                0.999
            ],
            "eps": 1e-8
            }
        },
    }  
     
    # Tokenize
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

    # preprocess_function
    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args,
            padding=padding,
            max_length=max_seq_length,
            truncation=True
        )

        if label_to_id is not None and "label" in examples:
            result["label"] = [
                (label_to_id[l] if l != -1 else -1)
                for l in examples["label"]
            ]
        return result

    # tokenized dataset
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names

    )
    
    # train / eval split
    inds = int((int(parser_params['training_set_pct']) / 100) * len(raw_datasets["train"]))
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

    # deepspeed initialize
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config_params=ds_config,
        model_parameters=model.parameters())
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # create Dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size, 
        shuffle=True,
        collate_fn=data_collator
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    
    # Training loop
    num_epochs = 1
    
    for epoch in range(num_epochs):
        model_engine.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # Move batch to GPU
            batch = {key: value.to(model.device) for key, value in batch.items()}
            
            # Forward pass
            outputs = model_engine(**batch)
            loss = outputs.loss / ds_config["gradient_accumulation_steps"]
            
            # Backward pass
            model_engine.backward(loss)
            
            # Optimizer step
            if (step + 1) % ds_config["gradient_accumulation_steps"] == 0:
                model_engine.step()
                
            total_loss += loss.item()
            
            if step % 2 == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")
                
        # evaluation loop
        model_engine.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {key: value.to(model.device) for key, value in batch.items()}

                outputs = model_engine(**batch)
                
                # not loss, logits
                logits = outputs.logits
                
                preds = logits[:, -1, :]
                
                # answer label
                labels = batch["labels"]
                
                preds = preds.cpu().numpy()
                labels = labels.cpu().numpy()

                all_preds.append(preds)
                all_labels.append(labels)
                
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        

        eval_results = compute_metrics(all_preds, all_labels)
        print(f"[Eval] Epoch {epoch + 1}: {eval_results}")
    
    print("Training Completed")


if __name__ == "__main__":
    main()