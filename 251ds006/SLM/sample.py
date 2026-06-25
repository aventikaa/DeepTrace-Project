import torch
import tiktoken

from model import GPT
from config import *

enc = tiktoken.get_encoding("gpt2")

model = GPT(
    vocab_size=50257,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

model.to(device)

prompt = "Once upon a time"

tokens = enc.encode(prompt)

x = torch.tensor(
    [tokens],
    device=device
)

out = model.generate(
    x,
    max_new_tokens=100,
    temperature=0.8
)

text = enc.decode(
    out[0].tolist()
)

print(text)