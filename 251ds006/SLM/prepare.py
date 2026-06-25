from datasets import load_dataset
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

print("Loading TinyStories...")
dataset = load_dataset("roneneldan/TinyStories")

train = dataset["train"]
val = dataset["validation"]

def tokenize(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)
    return {"ids": ids}

train = train.map(tokenize, remove_columns=["text"])
val = val.map(tokenize, remove_columns=["text"])

train_ids = []

for item in train:
    train_ids.extend(item["ids"])

val_ids = []

for item in val:
    val_ids.extend(item["ids"])

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile("train.bin")
val_ids.tofile("val.bin")

print("Done")
print("Train tokens:", len(train_ids))
print("Val tokens:", len(val_ids))
#print sample stories
print("\nExample Story:")
print(dataset["train"][0]["text"])