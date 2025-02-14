import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_encoding(seq_len=100, d_model=16):
    """Computes sinusoidal positional encoding."""
    pos = np.arange(seq_len)[:, np.newaxis]  # Shape: (seq_len, 1)
    i = np.arange(d_model // 2)[np.newaxis, :]  # Shape: (1, d_model//2)
    div_term = np.exp(-np.log(10000.0) * (2 * i / d_model))  # Scaling factor

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(pos * div_term)  # Apply sine to even indices
    pe[:, 1::2] = np.cos(pos * div_term)  # Apply cosine to odd indices
    return pe

# Generate encoding
seq_len = 64  # Number of positions
d_model = 16   # Embedding dimension
pe = sinusoidal_encoding(seq_len, d_model)

# Plot different dimensions
plt.figure(figsize=(10, 6))
for dim in range(0, d_model, 2):  # Select every second dimension
    plt.plot(pe[:, dim], label=f"Dim {dim}")

plt.xlabel("Position")
plt.ylabel("Encoding Value")
plt.title("Sinusoidal Positional Encoding")
plt.legend()
plt.show()
