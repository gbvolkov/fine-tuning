import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
from evaluate import load

dataset = "xsum"
raw_datasets = load_dataset(dataset, trust_remote_code=True)
metric = load("rouge")

