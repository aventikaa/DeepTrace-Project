import torch
import torch.nn as nn

class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd,
        n_head,
        n_layer,
        dropout
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding = nn.Embedding(
            block_size,
            n_embd
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer
        )

        self.ln = nn.LayerNorm(n_embd)

        self.head = nn.Linear(
            n_embd,
            vocab_size
        )

    def forward(self, idx):

        B, T = idx.shape

        pos = torch.arange(
            T,
            device=idx.device
        )

        tok = self.token_embedding(idx)

        pos = self.position_embedding(pos)

        x = tok + pos

        x = self.transformer(x)

        x = self.ln(x)

        logits = self.head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.7
    ):
        self.eval()

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size:]

            logits = self(idx_cond)

            logits = logits[:, -1, :]

            logits = logits / temperature

            probs = torch.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                [idx, next_token],
                dim=1
            )

        return idx