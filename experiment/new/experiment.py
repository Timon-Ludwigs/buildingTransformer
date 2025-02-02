import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate Fibonacci sequence data
def generate_fibonacci_data(batch_size, seq_length, normalize=True):
    x = torch.randint(1, 10, (batch_size, 2)).float()  # Small starting values
    for _ in range(seq_length - 2):
        next_term = x[:, -1:] + x[:, -2:-1]
        x = torch.cat([x, next_term], dim=1)

    y = x[:, 2:]  # Target sequence (next Fibonacci numbers)
    x = x[:, :-2]  # Input sequence

    if normalize:
        x = torch.log(x + 1)  # Apply log transform to avoid large numbers
        y = torch.log(y + 1)

    return x, y

# Sinusoidal positional encoding
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.register_buffer("positional_encoding", self._create_encodings(d_model, max_seq_length))

    def _create_encodings(self, d_model, max_seq_length):
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        encodings = torch.zeros(max_seq_length, d_model)
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)
        return encodings

    def forward(self, x):
        return self.positional_encoding[:x.size(1), :]

# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, d_model, max_seq_length, trainable_encoding=True):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)

        if trainable_encoding:
            self.positional_encoding = nn.Embedding(max_seq_length, d_model)
        else:
            self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_length)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=4,
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1).float()
        token_embeds = self.embedding(x)

        if isinstance(self.positional_encoding, nn.Embedding):
            pos_embeds = self.positional_encoding(torch.arange(x.size(1), device=x.device))
        else:
            pos_embeds = self.positional_encoding(x)

        x = token_embeds + pos_embeds
        x = self.encoder(x)
        return self.fc(x).squeeze(-1)  # Output shape: [batch, seq]

# Training loop
def train_model(model, epochs, train_seq_length, batch_size=32, label="Model"):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()  # Mean Absolute Error (better than MSE here)
    loss_history = []

    for epoch in range(epochs):
        x_train, y_train = generate_fibonacci_data(batch_size, train_seq_length)
        outputs = model(x_train)

        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        print(f"{label} - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model, loss_history

# Evaluation function
def evaluate_model(model, train_seq_length, test_seq_lengths, batch_size=32):
    model.eval()
    accuracies = {}

    with torch.no_grad():
        for seq_length in [train_seq_length] + test_seq_lengths:
            x_test, y_test = generate_fibonacci_data(batch_size, seq_length)
            outputs = model(x_test)

            y_test_exp = torch.exp(y_test) - 1  # Convert log back to real Fibonacci numbers
            outputs_exp = torch.exp(outputs) - 1

            mae = torch.abs(outputs_exp - y_test_exp).mean().item()
            accuracies[seq_length] = mae
            print(f"Test sequence length {seq_length}: MAE = {mae:.4f}")

    return accuracies

# Run experiments
d_model = 128
max_seq_length = 50
train_seq_length = 30
test_seq_lengths = [30, 35, 50]
epochs = 20

# Sinusoidal Encoding
print("Sinusoidal Positional Encoding")
sinusoidal_model = TransformerModel(d_model, max_seq_length, trainable_encoding=False)
sinusoidal_model, sinusoidal_loss = train_model(sinusoidal_model, epochs, train_seq_length, label="Sinusoidal")
sinusoidal_mae = evaluate_model(sinusoidal_model, train_seq_length, test_seq_lengths)

# Trainable Encoding
print("\nTrainable Positional Encoding")
trainable_model = TransformerModel(d_model, max_seq_length, trainable_encoding=True)
trainable_model, trainable_loss = train_model(trainable_model, epochs, train_seq_length, label="Trainable")
trainable_mae = evaluate_model(trainable_model, train_seq_length, test_seq_lengths)

# Plot Loss Curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), sinusoidal_loss, label="Sinusoidal Encoding", marker='o')
plt.plot(range(1, epochs + 1), trainable_loss, label="Trainable Encoding", marker='s')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves for Fibonacci Sequence Prediction")
plt.legend()
plt.grid()
plt.show()
