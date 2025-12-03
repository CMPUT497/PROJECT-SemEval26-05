import os
import sys
import json
import math
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Model configuration
LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # Highest parameter open-source model
# Alternative options if 405B is too large:
# "meta-llama/Llama-3.1-70B-Instruct"
# "meta-llama/Llama-3.3-70B-Instruct"
# "meta-llama/Llama-3.2-3B-Instruct"

MAX_LEN = 512  # Llama can handle longer sequences
BATCH_SIZE = 4  # Smaller batch size due to model size
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 20  # Fewer epochs since only training head
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRAD_CLIP = 1.0

def score_to_bin(t: float) -> int:
    """Map normalized t in [0,1] to bin 1..5."""
    if t <= 0.29:
        return 1
    if t <= 0.49:
        return 2
    if t <= 0.69:
        return 3
    if t <= 0.89:
        return 4
    return 5

class EndingDataset(Dataset):
    """Dataset for humor scoring with Llama-style prompting."""
    def __init__(self, data_list: List[Dict[str, Any]], tokenizer, max_len=MAX_LEN):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Precompute normalized target and bins
        for item in self.data:
            avg = float(item["avg_score"])
            t = avg / 5.0
            item["t"] = t
            item["bin"] = score_to_bin(t)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create a natural language prompt for Llama
        prompt = f"""Context: {item['precontext']}
Sentence: {item['sentence']}
Homonym: {item['homonym']}
Sense: {item['sense']}
Ending: {item['ending']}

Rate how plausible this homonym is in the sentence and ending from 1-5:"""
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "t": torch.tensor(item["t"], dtype=torch.float32),
            "bin": item["bin"],
            "meta": item,
        }


class FrozenLlamaScorer(nn.Module):
    """Frozen Llama model with trainable regression head."""
    def __init__(self, model_name=LLAMA_MODEL, hidden_dim=512):
        super().__init__()
        
        print(f"Loading frozen Llama model: {model_name}")
        # Load model in half precision to save memory
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Automatically distribute across GPUs if needed
            low_cpu_mem_usage=True
        )
        
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # Get hidden size from config
        hidden_size = self.backbone.config.hidden_size
        
        # Trainable regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    @torch.no_grad()
    def get_embeddings(self, input_ids, attention_mask):
        """Extract frozen embeddings from Llama."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state, take mean of non-padding tokens
        last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden]
        
        # Mean pooling over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask  # [batch, hidden]
        
        return pooled

    def forward(self, input_ids, attention_mask):
        # Get frozen embeddings
        embeddings = self.get_embeddings(input_ids, attention_mask)
        
        # Convert to float32 for head
        embeddings = embeddings.float()
        
        # Pass through trainable head
        logit = self.head(embeddings).squeeze(-1)  # [batch]
        
        return embeddings, logit


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    # Keep backbone in eval mode
    model.backbone.eval()
    
    total_loss = 0.0
    mse_loss_fn = nn.MSELoss()
    pbar = tqdm(dataloader, desc="train", leave=False)
    
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        t_targets = batch["t"].to(device)
        
        optimizer.zero_grad()

        embeddings, logits = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use MSE loss directly on normalized scores [0,1]
        probs = torch.sigmoid(logits)
        loss = mse_loss_fn(probs, t_targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), GRAD_CLIP)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    t_list = []
    logits_list = []
    metas = []
    
    pbar = tqdm(dataloader, desc="eval", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        t_targets = batch["t"].to(device)
        
        embeddings, logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits)

        preds.append(probs.detach().cpu())
        t_list.append(t_targets.detach().cpu())
        logits_list.append(logits.detach().cpu())
        metas.extend(batch["meta"])

    preds = torch.cat(preds).numpy()
    t_list = torch.cat(t_list).numpy()
    logits_list = torch.cat(logits_list).numpy()
    mse = mean_squared_error(t_list * 5.0, preds * 5.0)
    return {"mse_1_5": mse, "preds": preds, "targets": t_list, "logits": logits_list, "metas": metas}


def fit_platt(logits_train, targets_train):
    X = logits_train.reshape(-1, 1)
    y = (targets_train > 0.5).astype(int)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(X, y)
    a = lr.coef_[0][0]
    b = lr.intercept_[0]
    return a, b


def apply_platt(logits, a, b):
    logits_affine = a * logits + b
    probs = 1.0 / (1.0 + np.exp(-logits_affine))
    return probs


def run_training(train_data: List[Dict[str, Any]], dev_data: List[Dict[str, Any]]):
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL, use_fast=True)
    
    # Llama doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_ds = EndingDataset(train_data, tokenizer)
    dev_ds = EndingDataset(dev_data, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FrozenLlamaScorer(model_name=LLAMA_MODEL).to(DEVICE)

    # Only optimize the head parameters
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )

    best_dev_mse = float("inf")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        print(f"Train loss: {train_loss:.4f}")
        
        dev_res = evaluate(model, dev_loader, DEVICE)
        print(f"Dev MSE (1..5): {dev_res['mse_1_5']:.4f}")
        
        if dev_res["mse_1_5"] < best_dev_mse:
            best_dev_mse = dev_res["mse_1_5"]
            torch.save(model.head.state_dict(), "best_frozen_llama_head.pt")
            print(f"Saved new best model with MSE: {best_dev_mse:.4f}")

    # Load best model
    model.head.load_state_dict(torch.load("best_frozen_llama_head.pt"))
    print("\nLoaded best model.")

    # Platt calibration
    final_dev = evaluate(model, dev_loader, DEVICE)
    logits = final_dev["logits"]
    targets = final_dev["targets"]
    
    try:
        a, b = fit_platt(logits, targets)
        print(f"Platt params: a={a:.4f}, b={b:.4f}")
    except Exception as e:
        print(f"Platt fit failed: {e}")
        a, b = 1.0, 0.0

    return model, tokenizer, (a, b)


@torch.no_grad()
def predict_single(model, tokenizer, item: Dict[str, Any], device=DEVICE, platt_params=None):
    prompt = f"""Context: {item['precontext']}
        Sentence: {item['sentence']}
        Homonym: {item['homonym']}
        Sense: {item['sense']}
        Ending: {item['ending']}

        Rate how plausible this homonym is in the sentence and ending from 1-5:"""
    
    enc = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt"
    )
    
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    
    embeddings, logit = model(input_ids=input_ids, attention_mask=attention_mask)
    logit = logit.detach().cpu().numpy().item()
    prob = 1.0 / (1.0 + math.exp(-logit))
    
    if platt_params is not None:
        a, b = platt_params
        prob = 1.0 / (1.0 + math.exp(-(a * logit + b)))
    
    pred_float = prob * 5.0
    pred_int = int(round(pred_float))
    pred_int = max(1, min(5, pred_int))
    
    return {"prob": prob, "pred_float": pred_float, "pred_int": pred_int}


def load_json_data(json_file: str):
    with open(json_file, 'r', encoding='utf8') as f:
        file_dict = json.load(f)
    data = []
    for id, item in file_dict.items():
        data_instance = {
            "id": id,
            "homonym": item['homonym'],
            "precontext": item['precontext'],
            "sense": item['judged_meaning'],
            "ending": item['ending'],
            "sentence": item['sentence'],
            "avg_score": item['average']
        }
        data.append(data_instance)
    return data


def main():
    train_path = 'data/train.json'
    train_data = load_json_data(train_path)
    dev_path = 'data/dev.json'
    dev_data = load_json_data(dev_path)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Dev samples: {len(dev_data)}")
    
    model, tokenizer, platt_params = run_training(train_data, dev_data)
    
    # Generate predictions
    predictions = []
    for item in tqdm(dev_data, desc="Predicting"):
        pred = predict_single(model, tokenizer, item, DEVICE, platt_params=platt_params)
        predictions.append(pred)
    
    output_path = "predictions/method3_llm_v1_predictions.jsonl"
    os.makedirs("predictions", exist_ok=True)
    with open(output_path, "w", encoding="utf8") as outfile:
        for item, pred in zip(dev_data, predictions):
            entry = {"id": item["id"], "prediction": pred["pred_int"]}
            outfile.write(json.dumps(entry) + "\n")
    
    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
