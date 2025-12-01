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
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Model configuration
BERT_MODEL = "microsoft/deberta-v3-large"
MAX_LEN = 256
BATCH_SIZE = 16
LR_BACKBONE = 9e-6
LR_HEAD = 3e-4
WEIGHT_DECAY = 0.001
EPOCHS = 80
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
CONTRASTIVE_TEMPERATURE = 0.09
LAMBDA_CONTRAST = 1.0
GRAD_CLIP = 1.0

def score_to_bin(t: float) -> int:
    """Map normalized t in [0,1] to bin 1..5 according to your mapping."""
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
    """
    Expects data_list: list of dicts with keys:
      'precontext' (C), 'sentence' (S), 'sense' (M), 'homonym' (H), 'ending' (E), 'avg_score' (1..5)
    """
    def __init__(self, data_list: List[Dict[str, Any]], tokenizer, max_len=MAX_LEN):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len

        # precompute normalized target and bins
        for item in self.data:
            avg = float(item["avg_score"])
            t = avg / 5.0
            item["t"] = t
            item["bin"] = score_to_bin(t)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Build input text: [SENSE:M] C S [HOM] H [/HOM] [SEP] E
        # Note: BERT doesn't have custom sense tokens, so we'll use regular text
        sense_text = f"Sense: {item['sense']}"
        hom = item["homonym"]
        C = item["precontext"]
        S = item["sentence"]
        E = item["ending"]

        # Format: Sense: <sense> <precontext> <sentence> Homonym: <homonym> [SEP] <ending>
        text = f"{sense_text} {C} {S} Homonym: {hom}"
        ending_text = E
        
        # Use BERT tokenizer with proper segment handling
        encoding = self.tokenizer(
            text,
            ending_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "t": torch.tensor(item["t"], dtype=torch.float32),
            "bin": item["bin"],
            "meta": item,  # keep original for debugging/inference
        }


class BertScorer(nn.Module):
    def __init__(self, backbone_name=BERT_MODEL, hidden_dim=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        # pooling dim:
        hidden_size = self.backbone.config.hidden_size
        # small projection to embedding space (optional)
        self.proj = nn.Linear(hidden_size, hidden_size)
        # final head (linear from embedding to logit)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Get embeddings from BERT
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # Use [CLS] token embedding (first token) as pooled output
        pooled = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
        emb = self.proj(pooled)     # [batch, hidden_size]
        logit = self.head(emb).squeeze(-1)  # [batch]
        return emb, logit


def compute_contrastive_loss(embeddings: torch.Tensor, bins: List[int], temperature=0.07, device="cpu"):
    """
    embeddings: [B, D] (not normalized)
    bins: list of ints length B (1..5)
    Compute InfoNCE-style loss where positives are other samples in same bin.
    If a sample has no in-batch positive (i.e., unique bin), we skip it in the loss.
    """
    # normalize embeddings
    emb_norm = F.normalize(embeddings, dim=1)  # [B, D]
    sim_matrix = torch.matmul(emb_norm, emb_norm.t())  # [B, B] cosine similarities
    # scale
    sim_matrix = sim_matrix / temperature

    B = emb_norm.size(0)
    loss = torch.tensor(0.0, device=device)
    count = 0

    bins_tensor = torch.tensor(bins, device=device)
    for i in range(B):
        # positives: indices j != i with same bin
        pos_mask = (bins_tensor == bins_tensor[i]) & (torch.arange(B, device=device) != i)
        pos_idxs = torch.nonzero(pos_mask).squeeze(1)
        if pos_idxs.numel() == 0:
            continue  # no positives for this anchor in batch
        # compute numerator: sum exp(sim(i, pos))
        numerator = torch.exp(sim_matrix[i, pos_idxs]).sum()
        # denominator: sum over all j != i
        denom_mask = torch.arange(B, device=device) != i
        denom_idxs = torch.nonzero(denom_mask).squeeze(1)
        denominator = torch.exp(sim_matrix[i, denom_idxs]).sum()
        # loss_i = - log (numerator / denominator)
        loss_i = -torch.log(numerator / (denominator + 1e-12) + 1e-12)
        loss = loss + loss_i
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)
    return loss / count

def train_epoch(model, dataloader, optimizer, scheduler, device, lambda_contrast=LAMBDA_CONTRAST):
    model.train()
    total_loss = 0.0
    bce_loss_fn = nn.BCEWithLogitsLoss()
    pbar = tqdm(dataloader, desc="train", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        t_targets = batch["t"].to(device)
        bins = batch["bin"]  # list of ints
        optimizer.zero_grad()

        emb, logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # normalize embeddings for contrastive loss usage
        emb_norm = F.normalize(emb, dim=1)

        # BCE loss (logits -> sigmoid)
        loss_bce = bce_loss_fn(logits, t_targets)

        # contrastive loss
        loss_contrast = compute_contrastive_loss(emb, bins, temperature=CONTRASTIVE_TEMPERATURE, device=device)

        loss = loss_bce + lambda_contrast * loss_contrast

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        pbar.set_postfix({"loss": loss.item(), "bce": loss_bce.item(), "con": loss_contrast.item() if isinstance(loss_contrast, torch.Tensor) else 0.0})

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
        token_type_ids = batch["token_type_ids"].to(device)
        t_targets = batch["t"].to(device)
        emb, logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        probs = torch.sigmoid(logits)

        preds.append(probs.detach().cpu())
        t_list.append(t_targets.detach().cpu())
        logits_list.append(logits.detach().cpu())
        metas.extend(batch["meta"])

    preds = torch.cat(preds).numpy()
    t_list = torch.cat(t_list).numpy()
    logits_list = torch.cat(logits_list).numpy()
    mse = mean_squared_error(t_list * 5.0, preds * 5.0)  # compare on 1..5 scale
    return {"mse_1_5": mse, "preds": preds, "targets": t_list, "logits": logits_list, "metas": metas}

def fit_platt(logits_train, targets_train):
    # logits_train: numpy shape [N], targets in [0,1] floats
    # Fit logistic regression without regularization on single feature (logit)
    X = logits_train.reshape(-1, 1)
    y = (targets_train > 0.5).astype(int)  # we will train to separate >0.5 vs <=0.5 for calibration
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(X, y)
    a = lr.coef_[0][0]
    b = lr.intercept_[0]
    return a, b

def apply_platt(logits, a, b):
    logits_affine = a * logits + b
    probs = 1.0 / (1.0 + np.exp(-logits_affine))
    return probs

def run_training(train_data: List[Dict[str, Any]], dev_data: List[Dict[str, Any]]):
    # tokenizer + dataset + dataloader
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=True)
    train_ds = EndingDataset(train_data, tokenizer)
    dev_ds = EndingDataset(dev_data, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BertScorer(backbone_name=BERT_MODEL).to(DEVICE)

    # separate parameter groups
    backbone_params = list(model.backbone.parameters()) + list(model.proj.parameters())
    head_params = list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
    ])
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

    best_dev_mse = float("inf")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE, lambda_contrast=LAMBDA_CONTRAST)
        print("Train loss:", train_loss)
        dev_res = evaluate(model, dev_loader, DEVICE)
        print("Dev MSE (1..5):", dev_res["mse_1_5"])
        if dev_res["mse_1_5"] < best_dev_mse:
            best_dev_mse = dev_res["mse_1_5"]
            torch.save(model.state_dict(), "best_model_3.pt")
            # optionally save tokenizer

    # After training, load best model
    model.load_state_dict(torch.load("best_model_3.pt"))
    print("Loaded best model.")

    # collect dev logits/targets for optional Platt calibration
    # (we already computed dev_res above; compute full dev predictions)
    final_dev = evaluate(model, dev_loader, DEVICE)
    logits = final_dev["logits"]  # numpy
    targets = final_dev["targets"]
    # Fit Platt scaling (optional, simplistic)
    try:
        a, b = fit_platt(logits, targets)
        print("Platt params:", a, b)
    except Exception as e:
        print("Platt fit failed:", e)
        a, b = 1.0, 0.0

    return model, tokenizer, (a, b)

@torch.no_grad()
def predict_single(model, tokenizer, item: Dict[str, Any], device=DEVICE, platt_params=None):
    """
    item: dict with keys 'precontext','sentence','sense','homonym','ending'
    returns q (calibrated) and integer score 1..5
    """
    sense_text = f"Sense: {item['sense']}"
    hom = item["homonym"]
    C = item["precontext"]
    S = item["sentence"]
    E = item["ending"]
    
    text = f"{sense_text} {C} {S} Homonym: {hom}"
    ending_text = E
    
    enc = tokenizer(
        text,
        ending_text,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc["token_type_ids"].to(device)
    
    emb, logit = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
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
        data_instance = {}
        data_instance["id"] = id
        data_instance["homonym"] = item['homonym']
        data_instance["precontext"] = item['precontext']
        data_instance["sense"] = item['judged_meaning']
        data_instance["ending"] = item['ending']
        data_instance["sentence"] = item['sentence']
        data_instance["avg_score"] = item['average']
        data.append(data_instance)
    return data

def main():
    # Example toy data format (replace with your real data)
    train_path = 'data/train.json'
    train_data = load_json_data(train_path)
    dev_path = 'data/dev.json'
    dev_data = load_json_data(dev_path)
    
    model, tokenizer, platt_params = run_training(train_data, dev_data)
    
    # To get predictions in order of input for a dataset, just iterate over your list in order:
    predictions = []
    for item in dev_data:
        pred = predict_single(model, tokenizer, item, DEVICE, platt_params=platt_params)
        predictions.append(pred)
    
    output_path = "predictions/method3_predictions.JSONL"
    with open(output_path, "w", encoding="utf8") as outfile:
        for item, pred in zip(dev_data, predictions):
            entry = {"id": item["id"], "prediction": pred["pred_int"]}
            outfile.write(json.dumps(entry) + "\n")
  

if __name__ == "__main__":
    main()