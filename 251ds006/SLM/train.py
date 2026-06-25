import torch
import numpy as np
import torch.nn.functional as F

from model import GPT
from config import *

train_data = np.memmap(
    "train.bin",
    dtype=np.uint16,
    mode="r"
)

val_data = np.memmap(
    "val.bin",
    dtype=np.uint16,
    mode="r"
)

def get_batch(split):

    data = train_data if split == "train" else val_data

    ix = torch.randint(
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        torch.from_numpy(
            data[i:i+block_size].astype(np.int64)
        )
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(
            data[i+1:i+block_size+1].astype(np.int64)
        )
        for i in ix
    ])

    return x.to(device), y.to(device)

model = GPT(
    vocab_size=50257,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout
).to(device)
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_iters
)

print(
    "Initial LR:",
    optimizer.param_groups[0]['lr']
)

scheduler.step()

print(
    "New LR:",
    optimizer.param_groups[0]['lr']
)
best_val = 999999

train_losses = []
val_losses = []

@torch.no_grad()
def estimate_loss():

    model.eval()

    out = {}

    for split in ["train", "val"]:

        losses = []

        for _ in range(eval_iters):

            x, y = get_batch(split)

            logits = model(x)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            if iter % 10 == 0:
                print(f"iter {iter} loss {loss.item():.4f}")

            losses.append(loss.item())

        out[split] = sum(losses)/len(losses)

    model.train()

    return out

for iter in range(max_iters):

    x, y = get_batch("train")

    logits = model(x)

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if iter % eval_interval == 0:

        losses = estimate_loss()

        train_losses.append(
            losses["train"]
        )

        val_losses.append(
            losses["val"]
        )

        print(
            f"step {iter} "
            f"train {losses['train']:.4f} "
            f"val {losses['val']:.4f}"
        )

        if losses["val"] < best_val:

            best_val = losses["val"]

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

print("Training Complete")
x, y = get_batch("train")

print("\nSHIFT TEST")

print("x:", x[0][:10])

print("y:", y[0][:10])
import matplotlib.pyplot as plt

plt.plot(
    train_losses,
    label="train"
)

plt.plot(
    val_losses,
    label="val"
)

plt.legend()

plt.savefig(
    "loss_curve.png"
)

plt.show()