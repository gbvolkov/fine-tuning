import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
from evaluate import load
import nltk
import numpy as np
import torch
import os

DATASET="xsum"
MODEL="t5-small"
METRIC="rouge"
#Define prompt-tuning parameters
VIRTUAL_TOKENS = 20  # Number of virtual tokens for prompt-tuning
INIT_FROM_VOCAB = True

def get_last_checkpoint(output_dir):
    if os.path.isdir(output_dir):
        checkpoints = [folder for folder in os.listdir(output_dir) if folder.startswith("checkpoint")]
        if len(checkpoints) > 0:
            return os.path.join(output_dir, max(checkpoints))
    return None


def load_dataset_and_model(dataset, model_checkpoint, metric_name):
    #Load dataset and metric
    raw_datasets = load_dataset(dataset, trust_remote_code=True)
    metric = load(metric_name)

    #Load model and tokenizer
    output_dir = f"{model_checkpoint}-prompt-tuned-xsum"
    # Check if there's a checkpoint to resume from
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint is not None:
        model_checkpoint = last_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    return raw_datasets, metric, tokenizer, model

def setup_prompt_tuning(model, num_virtual_tokens, initialize_from_vocab):
    #Prepare the prompt embeddings
    prompt_embeddings = torch.nn.Embedding(num_virtual_tokens, model.config.d_model)
    if initialize_from_vocab:
        prompt_embeddings.weight.data = model.shared.weight[:num_virtual_tokens].clone().detach()

    #Freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False

    #Make prompt embeddings trainable
    prompt_embeddings.weight.requires_grad = True
    return prompt_embeddings

def set_device(model, prompt_embeddings):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Move model and prompt embeddings to device
    model = model.to(device)
    prompt_embeddings = prompt_embeddings.to(device)
    return model, prompt_embeddings

def set_model_forward(model, prompt_embeddings, num_virtual_tokens):
    ##Modify the model's forward pass to include prompt-tuning
    # Save the original forward method
    original_forward = model.forward
    def model_forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, 
                    decoder_input_ids=None, decoder_attention_mask=None, labels=None, **kwargs):
        # If we already have inputs_embeds, we're in a recursive call. Just add the prompt and return.
        if inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            prompt_embeds = prompt_embeddings.weight.repeat(batch_size, 1, 1)
            inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
            
            if attention_mask is not None:
                attention_mask = torch.cat([torch.ones(batch_size, num_virtual_tokens).to(self.device), attention_mask], dim=1)
            
            return original_forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                    decoder_input_ids=decoder_input_ids, 
                                    decoder_attention_mask=decoder_attention_mask,
                                    labels=labels, **kwargs)

        # If we have input_ids, convert them to embeddings
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            #inputs_embeds = self.encoder.embed_tokens(input_ids)
            #prompt_embeds = prompt_embeddings.weight.repeat(batch_size, 1, 1)
            #inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dsim=1)
            
            if attention_mask is not None:
                attention_mask = torch.cat([torch.ones(batch_size, num_virtual_tokens).to(self.device), attention_mask], dim=1)
            
            return original_forward(input_ids=input_ids, attention_mask=attention_mask,
                                    decoder_input_ids=decoder_input_ids, 
                                    decoder_attention_mask=decoder_attention_mask,
                                    labels=labels, **kwargs)

        # If we have neither, raise an error
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    #Apply the custom forward method
    model.forward = model_forward.__get__(model)


#Tokenize datasets
def tokenize_dataset(raw_datasets, tokenizer):
    #Preprocessing function
    max_input_length = 512
    max_target_length = 128
    def preprocess_function(examples):
        inputs = examples["document"]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    return tokenized_datasets

def setup_trainder(model, tokenized_datasets, tokenizer, metric):
    #Compute metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
        # Extract a few results
        result = {key: value * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    #Set up training arguments
    batch_size = 8
    model_name = MODEL.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"{model_name}-prompt-tuned-xsum",
        evaluation_strategy="steps",
        save_strategy="steps",  # Save checkpoints based on number of steps
        save_steps=500,  # Save a checkpoint every 1000 steps
        learning_rate=1e-3,  # Higher learning rate for prompt-tuning
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,  # Keep only the 3 most recent checkpoints
        num_train_epochs=10,  # Increase epochs as we're training fewer parameters
        predict_with_generate=True,
        fp16=True,
        push_to_hub=True,
        load_best_model_at_end=True,
        metric_for_best_model="rouge2",  # This will use the 'rouge2' score from compute_metrics
    )

    #Define data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    #Set up trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        #compute_metrics=lambda pred: metric.compute(predictions=pred.predictions, references=pred.label_ids, use_stemmer=True),
    )
    return trainer

def load_model_locally(path):
    """
    Load the model and tokenizer from a local directory.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    # Load the prompt embeddings
    prompt_embeddings = torch.nn.Embedding(20, model.config.d_model)  # Adjust 20 if you used a different number
    prompt_embeddings.load_state_dict(torch.load(f"{path}/prompt_embeddings.pt"))
    
    # Attach prompt embeddings to the model
    model.prompt_embeddings = prompt_embeddings
    
    # Recreate the custom forward method
    def model_forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0]
        prompt_embeds = self.prompt_embeddings.weight.repeat(batch_size, 1, 1)
        inputs_embeds = self.encoder.embed_tokens(input_ids)
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
        
        attention_mask = torch.cat([torch.ones(batch_size, 20).to(self.device), attention_mask], dim=1)
        
        return self.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

    model.forward = model_forward.__get__(model)
    
    return model, tokenizer

def summarize_text(model, tokenizer, text):
    """
    Use the model to summarize a given text.
    """
    inputs = tokenizer([text], max_length=512, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=100, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

#Train the model
def train_model(trainer):
    trainer.train()

raw_datasets, metric, tokenizer, model = load_dataset_and_model(DATASET, MODEL, METRIC)
prompt_embeddings  = setup_prompt_tuning(model, VIRTUAL_TOKENS, INIT_FROM_VOCAB)
model, prompt_embeddings = set_device(model, prompt_embeddings)
set_model_forward(model, prompt_embeddings, VIRTUAL_TOKENS)
tokenized_datasets = tokenize_dataset(raw_datasets, tokenizer)
trainer  = setup_trainder(model, tokenized_datasets, tokenizer, metric)

train_model(trainer)
trainer.save_model("t5-small-xsum-prompt-final")



