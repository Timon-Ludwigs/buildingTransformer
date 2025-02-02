import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import TranslationDataset
from experiment.model.transformer_experiment import TransformerModel
from src.utils.data_cleaning import clean_dataset

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    return logging.getLogger(__name__)

class EnhancedTrainer:
    def __init__(self, model, tokenizer, train_loader, val_loader, config, model_type=""):
        self.model = model.to(config["device"])
        self.tokenizer = tokenizer
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_type = model_type
        
        # Initialize tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.position_accuracies = []
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                (step + 1) ** -0.5, (step + 1) * config["warmup_steps"] ** -1.5
            ),
        )
        
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.logger = setup_logger()

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc=f"Training {self.model_type}"):
            source_ids = batch["source_ids"].to(self.config["device"])
            source_mask = batch["source_mask"].to(self.config["device"])
            target_ids = batch["target_ids"].to(self.config["device"])
            target_mask = batch["target_mask"].to(self.config["device"])
            labels = batch["labels"].to(self.config["device"])

            self.optimizer.zero_grad()
            output = self.model(source_ids, target_ids, source_mask, target_mask)
            loss = self.loss_fn(output.view(-1, output.size(-1)), labels.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0

        for batch in tqdm(self.val_loader, desc=f"Validating {self.model_type}"):
            source_ids = batch["source_ids"].to(self.config["device"])
            source_mask = batch["source_mask"].to(self.config["device"])
            target_ids = batch["target_ids"].to(self.config["device"])
            target_mask = batch["target_mask"].to(self.config["device"])
            labels = batch["labels"].to(self.config["device"])

            output = self.model(source_ids, target_ids, source_mask, target_mask)
            loss = self.loss_fn(output.view(-1, output.size(-1)), labels.view(-1))
            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss

    def analyze_position_sensitivity(self):
        """Analyze model performance at different positions in the sequence"""
        self.model.eval()
        position_correct = torch.zeros(self.config["max_len"]).to(self.config["device"])
        position_total = torch.zeros(self.config["max_len"]).to(self.config["device"])
        
        with torch.no_grad():
            for batch in self.val_loader:
                source_ids = batch["source_ids"].to(self.config["device"])
                source_mask = batch["source_mask"].to(self.config["device"])
                target_ids = batch["target_ids"].to(self.config["device"])
                target_mask = batch["target_mask"].to(self.config["device"])
                labels = batch["labels"].to(self.config["device"])

                output = self.model(source_ids, target_ids, source_mask, target_mask)
                predictions = output.argmax(dim=-1)
                
                # Calculate accuracy for each position
                for pos in range(min(self.config["max_len"], predictions.size(1))):
                    mask = target_mask[:, pos] == 1
                    if mask.any():
                        position_correct[pos] += (predictions[:, pos][mask] == labels[:, pos][mask]).sum()
                        position_total[pos] += mask.sum()
        
        position_accuracy = (position_correct / position_total.clamp(min=1)).cpu().numpy()
        self.position_accuracies.append(position_accuracy)
        return position_accuracy

    def test_robustness(self):
        """Test model robustness to different perturbations"""
        self.model.eval()
        results = {}
        
        test_cases = {
            'normal': lambda x: x,
            'shuffled_middle': self._shuffle_middle,
            'dropped_tokens': self._drop_random_tokens,
            'repeated_tokens': self._repeat_tokens
        }
        
        for test_name, perturbation in test_cases.items():
            total_loss = 0
            for batch in self.val_loader:
                source_ids = batch["source_ids"].to(self.config["device"])
                source_mask = batch["source_mask"].to(self.config["device"])
                target_ids = batch["target_ids"].to(self.config["device"])
                target_mask = batch["target_mask"].to(self.config["device"])
                labels = batch["labels"].to(self.config["device"])
                
                perturbed_source = perturbation(source_ids)
                
                with torch.no_grad():
                    output = self.model(perturbed_source, target_ids, source_mask, target_mask)
                    loss = self.loss_fn(output.view(-1, output.size(-1)), labels.view(-1))
                    total_loss += loss.item()
            
            results[test_name] = total_loss / len(self.val_loader)
        
        return results

    def _shuffle_middle(self, tensor, shuffle_ratio=0.5):
        # Shuffle a larger portion of the sequence
        result = tensor.clone()
        seq_len = tensor.size(1)
        if seq_len > 4:
            middle_start = int(seq_len * (0.5 - shuffle_ratio/2))
            middle_end = int(seq_len * (0.5 + shuffle_ratio/2))
            middle = result[:, middle_start:middle_end]
            idx = torch.randperm(middle_end - middle_start)
            result[:, middle_start:middle_end] = middle[:, idx]
        return result

    def _drop_random_tokens(self, tensor, drop_prob=0.2):
        mask = torch.rand(tensor.shape, device=tensor.device) > drop_prob
        result = tensor.clone()
        result[~mask] = self.tokenizer.pad_token_id
        return result

    def _repeat_tokens(self, tensor, repeat_prob=0.2):
        result = tensor.clone()
        for i in range(tensor.size(1)-1):
            if torch.rand(1).item() < repeat_prob:
                result[:, i+1] = result[:, i]
        return result

    def train(self):
        best_val_loss = float("inf")

        for epoch in range(self.config["num_epochs"]):
            self.logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']}")

            train_loss = self.train_epoch()
            self.logger.info(f"{self.model_type} Training Loss: {train_loss:.4f}")

            if (epoch + 1) % self.config["val_interval"] == 0:
                val_loss = self.validate()
                self.logger.info(f"{self.model_type} Validation Loss: {val_loss:.4f}")
                
                # Analyze position sensitivity every validation epoch
                pos_acc = self.analyze_position_sensitivity()
                self.logger.info(f"Position accuracy analyzed for epoch {epoch + 1}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(f"./experiment/checkpoints/best_model_{self.model_type}.pth")

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "position_accuracies": self.position_accuracies
        }, path)
        self.logger.info(f"Checkpoint saved at {path}")

def prepare_data(tokenizer, train_split="train[:10000]", val_split="validation[:4000]"):
    train_data = clean_dataset(load_dataset("wmt17", "de-en", split=train_split))
    val_data = clean_dataset(load_dataset("wmt17", "de-en", split=val_split))

    train_dataset = TranslationDataset(train_data, tokenizer)
    val_dataset = TranslationDataset(val_data, tokenizer)

    return train_dataset, val_dataset

def plot_enhanced_comparison(sin_trainer, learn_trainer, save_path="enhanced_comparison.png"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training and validation loss
    axes[0, 0].plot(sin_trainer.train_losses, label='Sinusoidal (train)')
    axes[0, 0].plot(sin_trainer.val_losses, label='Sinusoidal (val)')
    axes[0, 0].plot(learn_trainer.train_losses, label='Learnable (train)')
    axes[0, 0].plot(learn_trainer.val_losses, label='Learnable (val)')
    axes[0, 0].set_title('Loss Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Position sensitivity
    axes[0, 1].plot(sin_trainer.position_accuracies[-1], label='Sinusoidal')
    axes[0, 1].plot(learn_trainer.position_accuracies[-1], label='Learnable')
    axes[0, 1].set_title('Position-wise Accuracy')
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Robustness comparison
    sin_robust = sin_trainer.test_robustness()
    learn_robust = learn_trainer.test_robustness()
    
    x = np.arange(len(sin_robust))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, list(sin_robust.values()), width, label='Sinusoidal')
    axes[1, 0].bar(x + width/2, list(learn_robust.values()), width, label='Learnable')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(list(sin_robust.keys()), rotation=45)
    axes[1, 0].set_title('Robustness Comparison')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():

    trainer_config = {
        "batch_size": 64,
        "num_epochs": 10,
        "learning_rate": 0.05,  
        "warmup_steps": 4000,
        "val_interval": 1,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "max_len": 64
    }

    # Model configuration
    model_config = {
        "vocab_size": 50000,
        "input_dim": 128,
        "max_len": 64,
        "num_heads": 8,
        "num_encoder_layers": 8,
        "num_decoder_layers": 8,
        "feature_dim": 128,
        "dropout": 0.2
    }


    # Initialize tokenizer and prepare data (your existing code)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens(
        {"pad_token": "[PAD]", "bos_token": "[BOS]", "eos_token": "[EOS]", "unk_token": "[UNK]"}
    )
    model_config["vocab_size"] = len(tokenizer)

    train_dataset, val_dataset = prepare_data(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=trainer_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=trainer_config["batch_size"], shuffle=False)

    # Initialize models with different positional encodings
    sin_model = TransformerModel(**model_config, pos_encoding="sinusoidal")
    learn_model = TransformerModel(**model_config, pos_encoding="learnable")

    # Initialize enhanced trainers
    sin_trainer = EnhancedTrainer(sin_model, tokenizer, train_loader, val_loader, trainer_config, "sinusoidal")
    learn_trainer = EnhancedTrainer(learn_model, tokenizer, train_loader, val_loader, trainer_config, "learnable")

    # Train both models
    print("Training Sinusoidal Model...")
    sin_trainer.train()
    
    print("\nTraining Learnable Model...")
    learn_trainer.train()

    # Plot enhanced comparison
    plot_enhanced_comparison(sin_trainer, learn_trainer)

    # Test robustness
    print("\nTesting robustness...")
    sin_robust = sin_trainer.test_robustness()
    learn_robust = learn_trainer.test_robustness()
    
    print("\nSinusoidal Model Robustness:")
    for test_name, loss in sin_robust.items():
        print(f"{test_name}: {loss:.4f}")
        
    print("\nLearnable Model Robustness:")
    for test_name, loss in learn_robust.items():
        print(f"{test_name}: {loss:.4f}")

if __name__ == "__main__":
    main()