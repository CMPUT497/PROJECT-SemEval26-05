import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm


MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 60
LR = 2e-5
LAMBDA_CONS = 0.1
LAMBDA_RANK = 0.4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_PATH = "data/train.json"
DEV_PATH = "data/dev.json"
TEST_PATH = "data/test.json"
OUT_PATH_dev = "predictions/method6_dev.jsonl"
OUT_PATH_test = "predictions/method6_test.jsonl"

os.makedirs("predictions", exist_ok=True)


class HomonymDataset(Dataset):
    def __init__(self, path, tokenizer):
        with open(path) as f:
            raw = json.load(f)

        self.data = []
        for k, v in raw.items():
            v["id"] = k
            self.data.append(v)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        sentence = ex["sentence"].replace(
            ex["homonym"],
            f"<HOM> {ex['homonym']} </HOM>"
        )

        text = (
            ex["precontext"] + " [SEP] " +
            sentence + " [SEP] " +
            ex["judged_meaning"] + " [SEP] " +
            ex["ending"]
        )

        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "id": ex["id"],
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "ratings": torch.tensor(ex["choices"], dtype=torch.long),
            "mean": torch.tensor(ex["average"], dtype=torch.float)
        }


class OrdinalHead(nn.Module):
    def __init__(self, hidden_size, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)
        self.thresholds = nn.Parameter(torch.arange(1, num_classes).float())

    def forward(self, h):
        latent = self.fc(h)
        probs = torch.sigmoid(self.thresholds - latent)
        return probs, latent


class MultiAnnotatorModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.heads = nn.ModuleList([OrdinalHead(hidden) for _ in range(5)])

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        h = out.last_hidden_state[:, 0]
        probs, scores = [], []
        for head in self.heads:
            p, s = head(h)
            probs.append(p)
            scores.append(s.squeeze(-1))
        return probs, scores


def ordinal_loss(probs, targets):
    targets = targets.unsqueeze(-1)
    gt = (targets > torch.arange(1, 5, device=targets.device)).float()
    return F.binary_cross_entropy(probs, gt)

def pairwise_ranking_loss(pred, gold):
    loss = 0.0
    count = 0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            sign = torch.sign(gold[i] - gold[j])
            if sign != 0:
                loss += F.relu(-sign * (pred[i] - pred[j]))
                count += 1
    return loss / max(count, 1)

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()

        probs, scores = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE)
        )

        loss_ord = sum(
            ordinal_loss(probs[i], batch["ratings"][:, i].to(DEVICE))
            for i in range(5)
        )

        mean_score = torch.mean(torch.stack(scores), dim=0)

        loss_cons = sum(
            torch.mean((s - mean_score) ** 2)
            for s in scores
        )

        loss_rank = pairwise_ranking_loss(
            mean_score,
            batch["mean"].to(DEVICE)
        )

        loss = loss_ord + LAMBDA_CONS * loss_cons + LAMBDA_RANK * loss_rank
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def predict(model, loader):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            _, scores = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE)
            )
            mean_score = torch.mean(torch.stack(scores), dim=0)
            pred = torch.round(mean_score).clamp(1, 5).long()

            for i, pid in enumerate(batch["id"]):
                preds.append({
                    "id": pid,
                    "prediction": int(pred[i].item())
                })

    return preds

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(["<HOM>", "</HOM>"])

    train_ds = HomonymDataset(TRAIN_PATH, tokenizer)
    dev_ds = HomonymDataset(DEV_PATH, tokenizer)
    test_ds = HomonymDataset(TEST_PATH, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = MultiAnnotatorModel(MODEL_NAME)
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_epoch(model, train_loader, optimizer)

    preds = predict(model, dev_loader)
    with open(OUT_PATH_dev, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    print(f"\nDev Predictions written to {OUT_PATH_dev}")
    
    preds = predict(model, test_loader)
    with open(OUT_PATH_test, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    print(f"\ntest Predictions written to {OUT_PATH_test}")

if __name__ == "__main__":
    main()
