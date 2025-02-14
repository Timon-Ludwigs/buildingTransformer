import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import logging
import os
import wandb
from src.dataset import TranslationDataset
from src.model.transformer import TransformerModel
from src.utils.data_cleaning import clean_dataset
from sacrebleu import corpus_bleu

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    return logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, tokenizer, train_loader, val_loader, config):
        self.model = model.to(config["device"])
        self.tokenizer = tokenizer
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer and scheduler
        self.optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                (step + 1) ** -0.5, (step + 1) * config["warmup_steps"] ** -1.5
            ),
        )

        # Loss function
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.logger = setup_logger()

        # Initialize WandB
        wandb.init(
            project="transformer-training",  # Change to your WandB project name
            config=config,
        )
        wandb.watch(self.model, log="all", log_freq=100)

    def calculate_bleu(self, predictions, references):
        # Convert token IDs back to text
        decoded_preds = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
        decoded_refs = [[self.tokenizer.decode(ref, skip_special_tokens=True)] for ref in references]
        # Calculate BLEU score
        bleu = corpus_bleu(decoded_preds, decoded_refs)
        return bleu.score

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):
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

            # Log training loss for each batch
            wandb.log({"Training Loss (Batch)": loss.item()})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        predictions = []
        references = []

        for batch in tqdm(self.val_loader, desc="Validating"):
            source_ids = batch["source_ids"].to(self.config["device"])
            source_mask = batch["source_mask"].to(self.config["device"])
            target_ids = batch["target_ids"].to(self.config["device"])
            target_mask = batch["target_mask"].to(self.config["device"])
            labels = batch["labels"].to(self.config["device"])

            output = self.model(source_ids, target_ids, source_mask, target_mask)
            loss = self.loss_fn(output.view(-1, output.size(-1)), labels.view(-1))
            total_loss += loss.item()

            # Collect predictions and references for BLEU
            pred_ids = torch.argmax(output, dim=-1).tolist()
            predictions.extend(pred_ids)
            references.extend(labels.tolist())

        bleu_score = self.calculate_bleu(predictions, references)

        # Log BLEU to WandB
        wandb.log({"Validation Loss": total_loss / len(self.val_loader), "BLEU Score": bleu_score})

        return total_loss / len(self.val_loader)


    #def save_checkpoint(self, path="./checkpoints/best_model.pth"):
        #os.makedirs(os.path.dirname(path), exist_ok=True)
        #torch.save({"model_state_dict": self.model.state_dict()}, path)
        #self.logger.info(f"Checkpoint saved at {path}")
        #wandb.save(path, base_path="./checkpoints/")


    def train(self):
        best_val_loss = float("inf")

        for epoch in range(self.config["num_epochs"]):
            self.logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']}")

            train_loss = self.train_epoch()
            self.logger.info(f"Training Loss: {train_loss:.4f}")
            wandb.log({"Training Loss (Epoch)": train_loss, "Epoch": epoch + 1})

            if (epoch + 1) % self.config["val_interval"] == 0:
                val_loss = self.validate()
                self.logger.info(f"Validation Loss: {val_loss:.4f}")

                #if val_loss < best_val_loss:
                #    best_val_loss = val_loss
                #    self.save_checkpoint(path="./checkpoints/best_model.pth")


def prepare_data(tokenizer, train_split="train[:10000]", val_split="validation[:2000]"):
    train_data = clean_dataset(load_dataset("wmt17", "de-en", split=train_split))
    val_data = clean_dataset(load_dataset("wmt17", "de-en", split=val_split))

    train_dataset = TranslationDataset(train_data, tokenizer)
    val_dataset = TranslationDataset(val_data, tokenizer)

    return train_dataset, val_dataset

def main():
    # Initialize WandB run
    wandb.init(project="transformer-training")

    # Hyperparameters
    trainer_config = {
        "batch_size": 32,
        "num_epochs": 10,
        "learning_rate": 0.01,  
        "warmup_steps": 2000,
        "val_interval": 1,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens(
        {"pad_token": "[PAD]", "bos_token": "[BOS]", "eos_token": "[EOS]", "unk_token": "[UNK]"}
    )
    model_config["vocab_size"] = len(tokenizer)

    # Prepare data
    train_dataset, val_dataset = prepare_data(tokenizer, train_split="train[:10000]", val_split="validation[:4000]")
    train_loader = DataLoader(train_dataset, batch_size=trainer_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=trainer_config["batch_size"], shuffle=False)

    for sample in train_dataset:
        print(sample["source_ids"].max().item())
        print(sample["target_ids"].max().item())
        break

    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(f"Vocab size: {len(tokenizer)}")

    # Initialize model
    model = TransformerModel(**model_config)

    # Initialize and start training
    trainer = Trainer(model, tokenizer, train_loader, val_loader, trainer_config)
    trainer.train()

if __name__ == "__main__":
    main()
