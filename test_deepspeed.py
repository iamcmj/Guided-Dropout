import torch
import deepspeed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset

deepspeed.init_distributed("nccl")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "meta-llama/Llama-3.2-3B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                              config_params=ds_config,
                                              model_parameters=model.parameters())

# Load dataset
dataset_name = "wikitext"
subset = "wikitext-2-raw-v1"
dataset = load_dataset(dataset_name, subset, split="train")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(
    tokenized_dataset,
    shuffle=True,
    batch_size=micro_batch_size,
    collate_fn=data_collator,
)

# Training loop
num_epochs = 1
model_engine.train()

for epoch in range(num_epochs):
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

        # Log progress
        if step % 2 == 0:
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")

    print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(train_dataloader)}")