import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")

data = np.memmap(
    "train.bin",
    dtype=np.uint16,
    mode="r"
)

print("Total tokens:", len(data))

print("\nDecoded text:\n")

print(
    enc.decode(
        data[:200].tolist()
    )
)