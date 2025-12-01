import os
import sys
import json
import math
import random
import statistics
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

BERT_BACKBONE = "microsoft/deberta-v3-large"
MAX_LEN = 256
BATCH_SIZE = 16
LR_BACKBONE = 5e-6
LR_HEAD = 3e-4
WEIGHT_DECAY = 0.01
EPOCHS = 50
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
CONTRASTIVE_TEMPERATURE = 0.07
LAMBDA_CONTRAST = 0.5
GRAD_CLIP = 1.0
SAVE_PATH = "best_model_3v2.pt"
PREDICTIONS_PATH = "predictions/method3_v2_predictions.JSONL"
SPECIAL_TOKENS = {"additional_special_tokens": ["[HOM]", "[/HOM]", "[SENSE]", "[/SENSE]"]}
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class EndingDataset(Dataset):
    def __init__(self, data_list: List[Dict[str, Any]], tokenizer, max_len=MAX_LEN):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len

        for item in self.data:
            avg = float(item["avg_score"])
            t = avg / 5.0
            item["t"] = t
            item["bin"] = score_to_bin(t)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sense_text = f"[SENSE] {item['sense']} [/SENSE]"
        hom = item["homonym"]
        C = item["precontext"]
        S = item["sentence"]
        E = item["ending"]
        sentence_with_hom = S.replace(hom, f"[HOM] {hom} [/HOM]", 1)
        text = f"{sense_text} {C} {sentence_with_hom}"
        ending_text = E

        encoding = self.tokenizer(
            text,
            ending_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding.get("token_type_ids", torch.zeros_like(input_ids)).squeeze(0)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        hom_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        try:
            start = tokens.index("[HOM]")
            end = tokens.index("[/HOM]", start + 1)
            if end - start > 1:
                hom_mask[start + 1:end] = 1
        except ValueError:
            pass

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "t": torch.tensor(item["t"], dtype=torch.float32),
            "bin": item["bin"],
            "hom_mask": hom_mask,
            "meta": item,
        }


class DebertaScorer(nn.Module):
    def __init__(self, backbone_name=BERT_BACKBONE, hidden_dim=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size

        self.proj_cls = nn.Linear(hidden_size, hidden_size)
        self.proj_ctx = nn.Linear(hidden_size, hidden_size)
        self.proj_end = nn.Linear(hidden_size, hidden_size)
        self.proj_hom = nn.Linear(hidden_size, hidden_size)

        concat_size = hidden_size * 4
        self.head = nn.Sequential(
            nn.Linear(concat_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def masked_mean(self, x: torch.Tensor, mask: torch.Tensor):
        mask = mask.unsqueeze(-1).float()  # [B, L, 1]
        s = (x * mask).sum(dim=1)  # [B, H]
        denom = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        return s / denom

    def forward(self, input_ids, attention_mask, token_type_ids, hom_mask=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        last = outputs.last_hidden_state  # [B, L, H]
        cls_emb = last[:, 0, :]           # [B, H]

        seg0_mask = (token_type_ids == 0) & (attention_mask == 1)
        seg1_mask = (token_type_ids == 1) & (attention_mask == 1)

        ctx_pool = self.masked_mean(last, seg0_mask)
        end_pool = self.masked_mean(last, seg1_mask)

        if hom_mask is not None:
            hom_pool = self.masked_mean(last, hom_mask)
        else:
            hom_pool = torch.zeros_like(cls_emb)

        p_cls = self.proj_cls(cls_emb)
        p_ctx = self.proj_ctx(ctx_pool)
        p_end = self.proj_end(end_pool)
        p_hom = self.proj_hom(hom_pool)

        combined = torch.cat([p_cls, p_ctx, p_end, p_hom], dim=1)  # [B, 4H]
        raw = self.head(combined).squeeze(-1)  # [B]

        prob = torch.sigmoid(raw)  # normalized prediction in [0,1]
        # return combined embedding for contrastive, the final normalized prob, and raw logits
        return combined, prob, raw


def score_to_bin(t: float) -> int:
    if t <= 0.29:
        return 1
    if t <= 0.49:
        return 2
    if t <= 0.69:
        return 3
    if t <= 0.89:
        return 4
    return 5

def load_json_data_for_training(json_file: str):
    with open(json_file, "r", encoding="utf8") as f:
        file_dict = json.load(f)
    data = []
    for idstr, item in file_dict.items():
        data_instance = {
            "id": idstr,
            "homonym": item.get("homonym", ""),
            "precontext": item.get("precontext", ""),
            "sense": item.get("judged_meaning", ""),
            "ending": item.get("ending", ""),
            "sentence": item.get("sentence", ""),
            "avg_score": item.get("average", 3.0)
        }
        data.append(data_instance)
    return data

def compute_soft_contrastive(embeddings: torch.Tensor, targets: torch.Tensor, temperature=0.07, alpha=20.0):
    """
    embeddings: [B, D] (not necessarily normalized)
    targets: [B] floats normalized in [0,1]
    """
    if embeddings.size(0) < 2:
        return torch.tensor(0.0, device=embeddings.device)
    emb_norm = F.normalize(embeddings, dim=1)
    sims = torch.matmul(emb_norm, emb_norm.t()) / temperature  # [B, B]
    B = emb_norm.size(0)
    t = targets.view(B, 1)  # [B,1]
    diff = (t - t.t()).abs()  # [B,B]
    # weight matrix: closer targets => higher positive weight
    weights = torch.exp(-alpha * diff ** 2)  # [B,B]
    eye = torch.eye(B, device=embeddings.device)
    weights = weights * (1.0 - eye)  # zero diagonal
    numer = (weights * torch.exp(sims)).sum(dim=1)  # [B]
    denom = (torch.exp(sims) * (1.0 - eye)).sum(dim=1) + 1e-12
    loss_i = -torch.log(numer / denom + 1e-12)
    mask_valid = numer > 1e-12
    if mask_valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    return loss_i[mask_valid].mean()

def fit_linear_calibration(logits_train: np.ndarray, targets_train: np.ndarray, alpha=1.0):
    X = logits_train.reshape(-1, 1)
    y = targets_train
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    a = float(model.coef_[0])
    b = float(model.intercept_)
    return a, b

def apply_linear_cal(logits: np.ndarray, a: float, b: float):
    calibrated = a * logits + b
    calibrated = np.clip(calibrated, 0.0, 1.0)
    return calibrated

def train_epoch(model, dataloader, optimizer, scheduler, device, lambda_contrast=LAMBDA_CONTRAST):
    model.train()
    total_loss = 0.0
    mse_loss_fn = nn.MSELoss()
    pbar = tqdm(dataloader, desc="train", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        t_targets = batch["t"].to(device)
        hom_mask = batch["hom_mask"].to(device)

        optimizer.zero_grad()
        emb, probs, raw = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, hom_mask=hom_mask)
        loss_reg = mse_loss_fn(probs, t_targets)
        loss_contrast = compute_soft_contrastive(emb, t_targets, temperature=CONTRASTIVE_TEMPERATURE)
        loss = loss_reg + lambda_contrast * loss_contrast

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        pbar.set_postfix({"loss": loss.item(), "reg": loss_reg.item(), "con": float(loss_contrast.detach().cpu().item()) if isinstance(loss_contrast, torch.Tensor) else 0.0})

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

@torch.no_grad()
def evaluate_and_save_preds(model, dataloader, device, tokenizer, platt_params=None, out_path=PREDICTIONS_PATH):
    """
    Runs model on dataloader, returns mse on 1..5 scale, and writes predictions JSONL expected by evaluation script.
    Also returns raw logits and t targets arrays for calibration.
    """
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
        hom_mask = batch["hom_mask"].to(device)
        t_targets = batch["t"].to(device)

        emb, probs, raw = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, hom_mask=hom_mask)
        probs_np = probs.detach().cpu().numpy()
        raw_np = raw.detach().cpu().numpy()
        preds.append(probs_np)
        t_list.append(t_targets.detach().cpu().numpy())
        logits_list.append(raw_np)
        
        # === Normalize/collect meta items robustly ===
        batch_meta = batch.get("meta", None)
        if batch_meta is None:
            # nothing to add
            continue

        # batch_meta can be a list of dicts, list of strings, single dict, or single string.
        # Convert everything into dicts with "id" key and append to metas list in order.
        if isinstance(batch_meta, (list, tuple)):
            for m in batch_meta:
                if isinstance(m, dict):
                    metas.append(m)
                else:
                    # m might be an id string/int
                    metas.append({"id": str(m)})
        else:
            # single element
            m = batch_meta
            if isinstance(m, dict):
                metas.append(m)
            else:
                metas.append({"id": str(m)})

    if len(preds) == 0:
        return {"mse_1_5": float("nan"), "preds": np.array([]), "targets": np.array([]), "logits": np.array([]), "metas": metas, "predictions_filepath": out_path}

    preds = np.concatenate(preds, axis=0)
    t_list = np.concatenate(t_list, axis=0)
    logits_list = np.concatenate(logits_list, axis=0)

    # compute mse on 1..5 scale
    mse_1_5 = float(((t_list * 5.0 - preds * 5.0) ** 2).mean())

    # optionally calibrate raw logits with platt_params (linear)
    if platt_params is not None:
        a, b = platt_params
        preds_cal = apply_linear_cal(logits_list, a, b)
    else:
        # use probs directly
        preds_cal = preds

    # write predictions JSONL in the format expected by evaluation script:
    # each line: {"id": item_id (string?), "prediction": integer}
    with open(out_path, "w", encoding="utf8") as fout:
        for meta, p in zip(metas, preds_cal):
            pred_float = float(p * 5.0)
            pred_int = int(round(pred_float))
            pred_int = max(1, min(5, pred_int))
            entry = {"id": meta["id"], "prediction": pred_int}
            fout.write(json.dumps(entry) + "\n")

    return {"mse_1_5": mse_1_5, "preds": preds_cal, "targets": t_list, "logits": logits_list, "metas": metas, "predictions_filepath": out_path}

def get_standard_deviation(l):
    return statistics.stdev(l)

def get_average(l):
    return sum(l) / len(l)

def is_within_standard_deviation(prediction, labels):
    avg = get_average(labels)
    stdev = get_standard_deviation(labels)

    if (avg - stdev) < prediction < (avg + stdev):
        return True
    if abs(avg - prediction) < 1:
        return True
    return False

def spearman_evaluation_score(predictions_filepath: str, gold_data: dict):
    gold_list = ["-"] * len(gold_data)
    pred_list = ["-"] * len(gold_data)

    with open(predictions_filepath, "r") as f:
        pred_lines = f.readlines()

    for line in pred_lines:
        line = json.loads(line)
        gold_list[int(line["id"])] = gold_data[str(line["id"])]["average"]
        pred_list[int(line["id"])] = line["prediction"]

    corr, value = spearmanr(pred_list, gold_list)
    print(f"----------\nSpearman Correlation: {corr}\nSpearman p-Value: {value}")

def accuracy_within_standard_deviation_score(predictions_filepath, gold_data):
    with open(predictions_filepath, "r") as f:
        pred_lines = f.readlines()

    correct_guesses = 0
    wrong_guesses = 0

    for line in pred_lines:
        line = json.loads(line)
        labels = gold_data[str(line["id"])]["choices"]
        if is_within_standard_deviation(line["prediction"], labels):
            correct_guesses += 1
        else:
            wrong_guesses += 1

    print(f"----------\nAccuracy: {correct_guesses / (correct_guesses + wrong_guesses)} ({correct_guesses}/{correct_guesses+wrong_guesses})")

def run_training(train_json="data/train.json", dev_json="data/dev.json"):
    # Load tokenizer and model
    print("Loading tokenizer & model:", BERT_BACKBONE)
    tokenizer = AutoTokenizer.from_pretrained(BERT_BACKBONE, use_fast=True)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    # Datasets
    train_data = load_json_data_for_training(train_json)
    dev_data = load_json_data_for_training(dev_json)

    train_ds = EndingDataset(train_data, tokenizer, max_len=MAX_LEN)
    dev_ds = EndingDataset(dev_data, tokenizer, max_len=MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DebertaScorer(backbone_name=BERT_BACKBONE).to(DEVICE)
    # resize embeddings to account for added special tokens
    model.backbone.resize_token_embeddings(len(tokenizer))

    # Optimizer with separate groups
    backbone_params = list(model.backbone.parameters()) + list(model.proj_cls.parameters()) + list(model.proj_ctx.parameters()) + list(model.proj_end.parameters()) + list(model.proj_hom.parameters())
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
    ])
    total_steps = max(1, len(train_loader) * EPOCHS)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    best_dev_mse = float("inf")
    best_epoch = -1
    best_cal_params = None

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE, lambda_contrast=LAMBDA_CONTRAST)
        print(f"Train loss: {train_loss:.6f}")

        # Evaluate and save raw predictions (no calibration)
        dev_res = evaluate_and_save_preds(model, dev_loader, DEVICE, tokenizer, platt_params=None, out_path=PREDICTIONS_PATH)
        print(f"Dev MSE (1..5) (uncalibrated): {dev_res['mse_1_5']:.6f}")

        # Fit linear calibration on dev set logits -> t
        try:
            a, b = fit_linear_calibration(dev_res["logits"], dev_res["targets"], alpha=1.0)
            print(f"Calibration params (a, b): {a:.6f}, {b:.6f}")
        except Exception as e:
            print("Calibration failed:", e)
            a, b = 1.0, 0.0

        # Re-run predictions with calibrated logits to compute calibrated MSE
        dev_res_cal = evaluate_and_save_preds(model, dev_loader, DEVICE, tokenizer, platt_params=(a, b), out_path=PREDICTIONS_PATH)
        print(f"Dev MSE (1..5) (calibrated): {dev_res_cal['mse_1_5']:.6f}")

        # Run your supplied evaluation script's Spearman and Accuracy checks on predictions file
        # (It expects the gold JSON in data/dev.json with "choices" key per item)
        try:
            input_path = dev_json
            results = []
            with open(input_path, "r", encoding="utf8") as f:
                data = f.readlines()
                for line in data:
                    line = json.loads(line)
                    ids = line["id"]
                    pred = line["prediction"]
                    for id in ids:
                        output = {"id": id, "prediction": pred}
                        results.append(output)

            output_path = dev_json
            with open(output_path, "w", encoding="utf8") as fout:
                for item in results:
                    fout.write(json.dumps(item) + "\n")
            with open(dev_json, "r", encoding="utf8") as f:
                gold_data = json.load(f)
            print("Running Spearman & Accuracy evaluation using", PREDICTIONS_PATH)
            spearman_evaluation_score(PREDICTIONS_PATH, gold_data)
            accuracy_within_standard_deviation_score(PREDICTIONS_PATH, gold_data)
        except Exception as e:
            print("Evaluation script failed:", e)

        # Save best model by calibrated dev mse
        if dev_res_cal["mse_1_5"] < best_dev_mse:
            best_dev_mse = dev_res_cal["mse_1_5"]
            best_epoch = epoch + 1
            best_cal_params = (a, b)
            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer_added_tokens": SPECIAL_TOKENS,
                "calibration": best_cal_params
            }, SAVE_PATH)
            print(f"Best model updated (epoch {best_epoch}) -> saved to {SAVE_PATH}")

    print(f"\nTraining finished. Best dev MSE: {best_dev_mse:.6f} at epoch {best_epoch}")
    print("Best calibration params:", best_cal_params)

    # Load best model checkpoint and return objects
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, tokenizer, best_cal_params

@torch.no_grad()
def predict_single_item(model, tokenizer, item: Dict[str, Any], device=DEVICE, cal_params=None):
    sense_text = f"[SENSE] {item['sense']} [/SENSE]"
    hom = item["homonym"]
    C = item["precontext"]
    S = item["sentence"]
    E = item["ending"]
    if hom and hom in S:
        sentence_with_hom = S.replace(hom, f"[HOM] {hom} [/HOM]", 1)
    else:
        sentence_with_hom = f"{S} [HOM] {hom} [/HOM]"

    text = f"{sense_text} {C} {sentence_with_hom}"
    enc = tokenizer(text, E, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
    # create hom_mask similarly
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    hom_mask = torch.zeros_like(input_ids, dtype=torch.bool).squeeze(0)
    try:
        start = tokens.index("[HOM]")
        end = tokens.index("[/HOM]", start + 1)
        if end - start > 1:
            hom_mask[start + 1:end] = 1
    except ValueError:
        pass
    hom_mask = hom_mask.unsqueeze(0).to(device)

    emb, prob, raw = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, hom_mask=hom_mask)
    logit = float(raw.detach().cpu().numpy())
    if cal_params is not None:
        a, b = cal_params
        prob = float(apply_linear_cal(np.array([logit]), a, b)[0])
    else:
        prob = float(prob.detach().cpu().numpy().item())
    pred_float = prob * 5.0
    pred_int = int(round(pred_float))
    pred_int = max(1, min(5, pred_int))
    return {"prob": prob, "pred_float": pred_float, "pred_int": pred_int}


if __name__ == "__main__":
    # Basic CLI: allow training paths override
    train_file = "data/train.json"
    dev_file = "data/dev.json"
    if len(sys.argv) >= 3:
        train_file = sys.argv[1]
        dev_file = sys.argv[2]
    print("Train file:", train_file, "Dev file:", dev_file)
    model, tokenizer, cal_params = run_training(train_file, dev_file)
    
    input_path = PREDICTIONS_PATH
    results = []
    with open(input_path, "r", encoding="utf8") as f:
        data = f.readlines()
        for line in data:
            line = json.loads(line)
            ids = line["id"]
            pred = line["prediction"]
            for id in ids:
                output = {"id": id, "prediction": pred}
                results.append(output)

    output_path = PREDICTIONS_PATH
    with open(output_path, "w", encoding="utf8") as fout:
        for item in results:
            fout.write(json.dumps(item) + "\n")

    print("Done. Best model saved to:", SAVE_PATH)
    print("Best calibration params:", cal_params)
    print("Final predictions saved to:", PREDICTIONS_PATH)
