# Import necessary libraries
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import get_peft_model, LoraConfig, TaskType

# Load dataset and tokenizer
model_checkpoint = "t5-small"
dataset_name = "xsum"
raw_datasets = load_dataset(dataset_name)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Preprocessing function
max_input_length = 1024
max_target_length = 128
prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize datasets
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# Load base model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['q', 'v']
)

# Wrap the model with LoRA
model = get_peft_model(model, peft_config)

# Set up training arguments
batch_size = 8
args = Seq2SeqTrainingArguments(
    f"{model_checkpoint}-finetuned-xsum-lora",
    evaluation_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)

# Set up data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Set up trainer
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the LoRA model
model.save_pretrained("t5-small-xsum-lora")
