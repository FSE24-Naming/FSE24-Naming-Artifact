from transformers import BertTokenizer, AutoModel, Trainer, TrainingArguments
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import json

with open("./vectorizer/pytorch/layer_name_data.json", "r") as f:
    sentences = json.load(f)

tokenizer = BertTokenizer("./vectorizer/pytorch/vocab.txt")

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

print(inputs)

dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])

print(dataset)

data_loader = DataLoader(dataset, batch_size=32)

model = AutoModel.from_pretrained('bert-base-uncased')

training_args = TrainingArguments(
    output_dir="./vectorizer/pytorch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./vectorizer/pytorch")