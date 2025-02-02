import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ========== 1. DATA GENERATION ==========
def generate_synthetic_data(batch_size, seq_length, vocab_size=100):
    """
    Generates synthetic sequences of integers and assigns a classification label.
    Label is a simple sum-based threshold classification.
    """
    x = torch.randint(1, vocab_size, (batch_size, seq_length))  # Random tokens
    y = (x.sum(dim=1) % 2).long()  # Binary label based on sum
    return x, y

# ========== 2. POSITONAL ENCODINGS ==========
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :].to(x.device)

class TrainablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]

# ========== 3. TRANSFORMER MODEL ==========
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len, use_sinusoidal=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len) if use_sinusoidal else TrainablePositionalEncoding(d_model, max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 2)  # Binary classification

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

# ========== 4. PERTURBATION FUNCTIONS ==========
def middle_sequence_shuffling(batch):
    """ Shuffle 50% of the sequence around the middle """
    batch_size, seq_length = batch.shape
    mid = seq_length // 2
    quarter = seq_length // 4
    start, end = mid - quarter, mid + quarter

    shuffled_batch = batch.clone()
    for i in range(batch_size):
        shuffled_batch[i, start:end] = shuffled_batch[i, start:end][torch.randperm(quarter * 2)]
    return shuffled_batch

def random_token_dropping(batch, pad_token=0, drop_prob=0.2):
    """ Randomly replace 20% of tokens with padding tokens """
    mask = torch.rand(batch.shape) < drop_prob
    batch[mask] = pad_token
    return batch

def token_repetition(batch, repeat_prob=0.2):
    """ Randomly duplicate 20% of tokens """
    batch_size, seq_length = batch.shape
    for i in range(batch_size):
        indices = torch.randperm(seq_length - 1)[:int(seq_length * repeat_prob)]
        for idx in indices:
            batch[i, idx + 1] = batch[i, idx]
    return batch

# ========== 5. TRAINING FUNCTION ==========
def train_model(model, train_data, test_data, epochs=5, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        x_train, y_train = train_data
        optimizer.zero_grad()
        y_pred = model(x_train).squeeze()
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    x_test, y_test = test_data
    with torch.no_grad():
        y_pred = model(x_test).argmax(dim=1)
        accuracy = (y_pred == y_test).float().mean().item()
    return accuracy

# ========== 6. EXPERIMENT SETUP ==========
batch_size, seq_length, vocab_size = 1000, 20, 100
d_model, num_heads, num_layers, max_len = 64, 4, 2, 100

# Create dataset
x_train, y_train = generate_synthetic_data(batch_size, seq_length, vocab_size)
x_test, y_test = generate_synthetic_data(200, seq_length, vocab_size)

# Define models
sinusoidal_model = TransformerClassifier(vocab_size, d_model, num_heads, num_layers, max_len, use_sinusoidal=True)
trainable_model = TransformerClassifier(vocab_size, d_model, num_heads, num_layers, max_len, use_sinusoidal=False)

# Train models
print("\nTraining Sinusoidal Model...")
sinusoidal_accuracy = train_model(sinusoidal_model, (x_train, y_train), (x_test, y_test))

print("\nTraining Trainable Model...")
trainable_accuracy = train_model(trainable_model, (x_train, y_train), (x_test, y_test))

# ========== 7. PERTURBATION TESTING ==========
def evaluate_perturbations(model, x_test, y_test, name):
    perturbations = {
        "Original": x_test,
        "Shuffled": middle_sequence_shuffling(x_test.clone()),
        "Token Dropped": random_token_dropping(x_test.clone(), pad_token=0),
        "Token Repeated": token_repetition(x_test.clone())
    }
    
    results = {}
    model.eval()
    with torch.no_grad():
        for perturbation, x_perturbed in perturbations.items():
            y_pred = model(x_perturbed).argmax(dim=1)
            accuracy = (y_pred == y_test).float().mean().item()
            results[perturbation] = accuracy
            print(f"{name} - {perturbation} Accuracy: {accuracy:.4f}")
    return results

print("\nEvaluating Sinusoidal Model...")
sinusoidal_results = evaluate_perturbations(sinusoidal_model, x_test, y_test, "Sinusoidal")

print("\nEvaluating Trainable Model...")
trainable_results = evaluate_perturbations(trainable_model, x_test, y_test, "Trainable")

# ========== 8. VISUALIZATION ==========
plt.figure(figsize=(8,5))
sns.barplot(x=list(sinusoidal_results.keys()), y=list(sinusoidal_results.values()), label="Sinusoidal", color="blue", alpha=0.6)
sns.barplot(x=list(trainable_results.keys()), y=list(trainable_results.values()), label="Trainable", color="red", alpha=0.6)
plt.ylabel("Accuracy")
plt.title("Impact of Perturbations on Transformer Models")
plt.legend()
plt.show()
