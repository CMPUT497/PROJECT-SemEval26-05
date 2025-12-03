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
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Model configuration
LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # Good balance of size and performance
# Alternative options:
# "meta-llama/Llama-3.3-70B-Instruct"
# "meta-llama/Llama-3.1-8B-Instruct" (for faster experimentation)

MAX_LEN = 512
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 16
LR = 2e-4
WEIGHT_DECAY = 0.01
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRAD_CLIP = 1.0

# LoRA hyperparameters
LORA_R = 16  # Rank
LORA_ALPHA = 32  # Scaling factor (typically 2*r)
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

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
        
        # Create a structured prompt for Llama
        prompt = f"""
            Analyze this homonym in the sentenceand rate its plausibility on a scale of 1-5.

            Context: {item['precontext']}
            Sentence: {item['sentence']}
            Homonym: {item['homonym']}
            Intended Sense: {item['sense']}
            Ending: {item['ending']}

            Plausibility Rating:
        """
        
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


class LoraLlamaScorer(nn.Module):
    """Llama model with LoRA adapters and regression head."""
    def __init__(self, model_name=LLAMA_MODEL, hidden_dim=512, use_8bit=True):
        super().__init__()
        
        print(f"Loading Llama model with LoRA: {model_name}")
        
        # Load model with quantization for memory efficiency
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True
        }
        
        if use_8bit:
            load_kwargs["load_in_8bit"] = True
        
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        # Prepare model for LoRA training
        if use_8bit:
            self.backbone = prepare_model_for_kbit_training(self.backbone)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        
        # Apply LoRA to the model
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()
        
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

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state with mean pooling
        last_hidden = outputs.hidden_states[-1]
        
        # Mean pooling over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # Convert to float32 for head
        pooled = pooled.float()
        
        # Pass through regression head
        logit = self.head(pooled).squeeze(-1)
        
        return pooled, logit


def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps=GRADIENT_ACCUMULATION_STEPS):
    model.train()
    total_loss = 0.0
    mse_loss_fn = nn.MSELoss()
    pbar = tqdm(dataloader, desc="train", leave=False)
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        t_targets = batch["t"].to(device)
        
        embeddings, logits = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use MSE loss on normalized scores [0,1]
        probs = torch.sigmoid(logits)
        loss = mse_loss_fn(probs, t_targets)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps * input_ids.size(0)
        pbar.set_postfix({"loss": loss.item() * accumulation_steps})

    # Handle remaining batches
    if len(dataloader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()

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

    model = LoraLlamaScorer(model_name=LLAMA_MODEL).to(DEVICE)

    # Optimize both LoRA parameters and head
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    
    # Calculate steps accounting for gradient accumulation
    steps_per_epoch = len(train_loader) // GRADIENT_ACCUMULATION_STEPS
    total_steps = steps_per_epoch * EPOCHS
    
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
            # Save LoRA adapters and head
            model.backbone.save_pretrained("best_lora_model")
            torch.save(model.head.state_dict(), "best_lora_head.pt")
            print(f"Saved new best model with MSE: {best_dev_mse:.4f}")

    # Load best model
    model.backbone.load_adapter("best_lora_model")
    model.head.load_state_dict(torch.load("best_lora_head.pt"))
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
    prompt = f"""
            Analyze this homonym in the sentenceand rate its plausibility on a scale of 1-5.

            Context: {item['precontext']}
            Sentence: {item['sentence']}
            Homonym: {item['homonym']}
            Intended Sense: {item['sense']}
            Ending: {item['ending']}

            Plausibility Rating:
        """
    
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
    
    output_path = "predictions/method3_llm_v2_predictions.jsonl"
    os.makedirs("predictions", exist_ok=True)
    with open(output_path, "w", encoding="utf8") as outfile:
        for item, pred in zip(dev_data, predictions):
            entry = {"id": item["id"], "prediction": pred["pred_int"]}
            outfile.write(json.dumps(entry) + "\n")
    
    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()