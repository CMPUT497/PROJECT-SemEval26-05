import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import math


@dataclass
class WSDInstance:
    """Single WSD instance with all components."""
    id: str
    homonym: str
    judged_meaning: str
    precontext: str
    sentence: str
    ending: str
    example_sentence: str
    average_score: float
    all_glosses: List[str]  # All possible meanings for this homonym


class WSDDataset(Dataset):
    """Dataset that provides multiple views of the same data for different objectives."""
    
    def __init__(self, instances: List[WSDInstance], tokenizer, max_len=256):
        self.instances = instances
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        inst = self.instances[idx]
        
        # 1. Context with marked homonym for ConSeC-style processing
        marked_sentence = inst.sentence.replace(
            inst.homonym, 
            f"<homonym> {inst.homonym} </homonym>"
        )
        full_context = f"{inst.precontext} {marked_sentence} {inst.ending}".strip()
        
        # 2. Narrative coherence text (for LLM-judge style)
        narrative_text = f"Context: {inst.precontext} **{inst.sentence}** {inst.ending} Meaning: {inst.judged_meaning}"
        
        # 3. Sense + Context + Ending (for Method3 style)
        sense_context_text = f"Sense: {inst.judged_meaning}"
        ending_text = f"{inst.precontext} {inst.sentence}"
        continuation_text = inst.ending
        
        # 4. Example sentence for similarity
        example_text = inst.example_sentence
        gloss_text = inst.judged_meaning
        
        # 5. All glosses for ranking
        all_glosses_text = " [SEP] ".join(inst.all_glosses)
        
        # Tokenize different views
        # View 1: Context with marked homonym
        context_enc = self.tokenizer(
            full_context,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        # View 2: Narrative coherence
        narrative_enc = self.tokenizer(
            narrative_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        # View 3: Sense + Ending pair
        sense_ending_enc = self.tokenizer(
            sense_context_text,
            continuation_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        # View 4: Context vs Example similarity
        context_example_enc = self.tokenizer(
            full_context,
            example_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        # View 5: Context vs Gloss similarity
        context_gloss_enc = self.tokenizer(
            full_context,
            gloss_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Normalize score to [0, 1]
        normalized_score = (inst.average_score - 1) / 4.0
        
        return {
            'context_input_ids': context_enc['input_ids'].squeeze(0),
            'context_attention_mask': context_enc['attention_mask'].squeeze(0),
            
            'narrative_input_ids': narrative_enc['input_ids'].squeeze(0),
            'narrative_attention_mask': narrative_enc['attention_mask'].squeeze(0),
            
            'sense_ending_input_ids': sense_ending_enc['input_ids'].squeeze(0),
            'sense_ending_attention_mask': sense_ending_enc['attention_mask'].squeeze(0),
            'sense_ending_token_type_ids': sense_ending_enc['token_type_ids'].squeeze(0),
            
            'context_example_input_ids': context_example_enc['input_ids'].squeeze(0),
            'context_example_attention_mask': context_example_enc['attention_mask'].squeeze(0),
            'context_example_token_type_ids': context_example_enc['token_type_ids'].squeeze(0),
            
            'context_gloss_input_ids': context_gloss_enc['input_ids'].squeeze(0),
            'context_gloss_attention_mask': context_gloss_enc['attention_mask'].squeeze(0),
            'context_gloss_token_type_ids': context_gloss_enc['token_type_ids'].squeeze(0),
            
            'score': torch.tensor(normalized_score, dtype=torch.float32),
            'id': inst.id
        }


class MultiTaskWSDModel(nn.Module):
    """
    Unified model that learns from multiple objectives:
    1. ConSeC-style: Context-sense matching
    2. LLM-Judge: Narrative coherence scoring
    3. Method3: Contrastive learning with ending effects
    4. OHPT: Cross-lingual similarity (adapted to monolingual)
    """
    
    def __init__(self, model_name="microsoft/deberta-v3-base", hidden_dim=768):
        super().__init__()
        
        # Shared encoder backbone
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = self.encoder.config
        hidden_size = self.config.hidden_size
        
        # Add special tokens
        self.special_tokens = ["<homonym>", "</homonym>"]
        
        # Task-specific heads
        
        # 1. ConSeC Head: Context-sense matching
        self.consec_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.consec_classifier = nn.Linear(hidden_dim, 1)
        
        # 2. Narrative Coherence Head (LLM-Judge style)
        self.narrative_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 3. Contrastive Embedding Head (Method3 style)
        self.contrastive_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ending_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 4. Similarity Head (OHPT style)
        self.similarity_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(4, 32),  # 4 task predictions
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
        # Learnable task weights
        self.task_weights = nn.Parameter(torch.ones(4))
    
    def encode(self, input_ids, attention_mask, token_type_ids=None):
        """Shared encoding function."""
        if token_type_ids is not None:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        return outputs
    
    def forward_consec(self, input_ids, attention_mask):
        """ConSeC-style forward: context-sense matching."""
        outputs = self.encode(input_ids, attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        projected = self.consec_projector(cls_embedding)
        logit = self.consec_classifier(projected)
        return torch.sigmoid(logit.squeeze(-1)), projected
    
    def forward_narrative(self, input_ids, attention_mask):
        """LLM-Judge style: narrative coherence."""
        outputs = self.encode(input_ids, attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        logit = self.narrative_head(cls_embedding)
        return torch.sigmoid(logit.squeeze(-1))
    
    def forward_contrastive(self, input_ids, attention_mask, token_type_ids):
        """Method3 style: contrastive with ending."""
        outputs = self.encode(input_ids, attention_mask, token_type_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        embedding = self.contrastive_projector(cls_embedding)
        logit = self.ending_scorer(embedding)
        return torch.sigmoid(logit.squeeze(-1)), embedding
    
    def forward_similarity(self, input_ids, attention_mask, token_type_ids):
        """OHPT style: similarity-based scoring."""
        outputs = self.encode(input_ids, attention_mask, token_type_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        embedding = self.similarity_projector(cls_embedding)
        # Compute similarity as normalized dot product
        normalized = F.normalize(embedding, dim=-1)
        # Self-similarity score (in batch, we'll use cross-batch similarity)
        similarity = torch.sum(normalized * normalized, dim=-1)
        return similarity
    
    def forward(self, batch):
        """Forward pass through all tasks."""
        # Task 1: ConSeC
        consec_pred, consec_emb = self.forward_consec(
            batch['context_input_ids'],
            batch['context_attention_mask']
        )
        
        # Task 2: Narrative
        narrative_pred = self.forward_narrative(
            batch['narrative_input_ids'],
            batch['narrative_attention_mask']
        )
        
        # Task 3: Contrastive
        contrastive_pred, contrast_emb = self.forward_contrastive(
            batch['sense_ending_input_ids'],
            batch['sense_ending_attention_mask'],
            batch['sense_ending_token_type_ids']
        )
        
        # Task 4: Similarity (Context-Example)
        sim_context_example = self.forward_similarity(
            batch['context_example_input_ids'],
            batch['context_example_attention_mask'],
            batch['context_example_token_type_ids']
        )
        
        # Task 5: Similarity (Context-Gloss)
        sim_context_gloss = self.forward_similarity(
            batch['context_gloss_input_ids'],
            batch['context_gloss_attention_mask'],
            batch['context_gloss_token_type_ids']
        )
        
        # Combine similarities
        similarity_pred = (sim_context_example + sim_context_gloss) / 2
        
        # Stack predictions
        task_preds = torch.stack([
            consec_pred,
            narrative_pred,
            contrastive_pred,
            similarity_pred
        ], dim=1)
        
        # Apply learned weights
        weights = F.softmax(self.task_weights, dim=0)
        weighted_preds = task_preds * weights.unsqueeze(0)
        
        # Final fusion
        final_score = self.fusion(weighted_preds)
        final_score = torch.sigmoid(final_score.squeeze(-1))
        
        return {
            'final_score': final_score,
            'consec_pred': consec_pred,
            'narrative_pred': narrative_pred,
            'contrastive_pred': contrastive_pred,
            'similarity_pred': similarity_pred,
            'consec_emb': consec_emb,
            'contrast_emb': contrast_emb,
            'task_weights': weights
        }


def compute_contrastive_loss(embeddings, scores, temperature=0.07):
    """
    Contrastive loss: similar scores should have similar embeddings.
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=-1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.t()) / temperature
    
    # Create target similarity based on score proximity
    score_diff = torch.abs(scores.unsqueeze(0) - scores.unsqueeze(1))
    target_sim = 1.0 - (score_diff / 4.0)  # Normalize to [0, 1]
    
    # InfoNCE-style loss
    log_probs = F.log_softmax(sim_matrix, dim=-1)
    loss = -torch.mean(torch.sum(target_sim * log_probs, dim=-1))
    
    return loss


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch with multi-task learning."""
    model.train()
    
    total_loss = 0
    total_mse = 0
    total_contrastive = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch in pbar:
        # Move batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch)
        
        # Main regression loss
        mse_loss = F.mse_loss(outputs['final_score'], batch['score'])
        
        # Task-specific losses
        consec_loss = F.mse_loss(outputs['consec_pred'], batch['score'])
        narrative_loss = F.mse_loss(outputs['narrative_pred'], batch['score'])
        contrastive_loss = F.mse_loss(outputs['contrastive_pred'], batch['score'])
        similarity_loss = F.mse_loss(outputs['similarity_pred'], batch['score'])
        
        # Contrastive embedding loss
        contrast_emb_loss = compute_contrastive_loss(
            outputs['contrast_emb'], 
            batch['score'],
            temperature=0.07
        )
        
        # Combined loss
        task_loss = (consec_loss + narrative_loss + 
                    contrastive_loss + similarity_loss) / 4
        
        total_loss_value = (
            0.5 * mse_loss +           # Main prediction
            0.3 * task_loss +          # Task-specific
            0.2 * contrast_emb_loss    # Contrastive regularization
        )
        
        total_loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += total_loss_value.item()
        total_mse += mse_loss.item()
        total_contrastive += contrast_emb_loss.item()
        
        pbar.set_postfix({
            'loss': f'{total_loss_value.item():.4f}',
            'mse': f'{mse_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_contrastive = total_contrastive / len(dataloader)
    
    return avg_loss, avg_mse, avg_contrastive


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_ids = []
    
    task_preds = {
        'consec': [],
        'narrative': [],
        'contrastive': [],
        'similarity': []
    }
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        outputs = model(batch)
        
        all_preds.extend(outputs['final_score'].cpu().numpy())
        all_targets.extend(batch['score'].cpu().numpy())
        all_ids.extend(batch['id'])
        
        task_preds['consec'].extend(outputs['consec_pred'].cpu().numpy())
        task_preds['narrative'].extend(outputs['narrative_pred'].cpu().numpy())
        task_preds['contrastive'].extend(outputs['contrastive_pred'].cpu().numpy())
        task_preds['similarity'].extend(outputs['similarity_pred'].cpu().numpy())
    
    # Convert back to 1-5 scale
    all_preds = np.array(all_preds) * 4 + 1
    all_targets = np.array(all_targets) * 4 + 1
    
    # Compute metrics
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    pearson, _ = pearsonr(all_targets, all_preds)
    spearman, _ = spearmanr(all_targets, all_preds)
    
    # Task-specific metrics
    task_metrics = {}
    for task_name, preds in task_preds.items():
        preds = np.array(preds) * 4 + 1
        task_metrics[task_name] = {
            'mse': mean_squared_error(all_targets, preds),
            'mae': mean_absolute_error(all_targets, preds)
        }
    
    return {
        'mse': mse,
        'mae': mae,
        'pearson': pearson,
        'spearman': spearman,
        'predictions': all_preds,
        'targets': all_targets,
        'ids': all_ids,
        'task_metrics': task_metrics
    }


def load_data(json_path: str) -> List[WSDInstance]:
    """Load data from JSON file."""
    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    
    # First pass: collect all glosses per homonym
    homonym_glosses = {}
    for item in data.values():
        homonym = item['homonym']
        gloss = item['judged_meaning']
        
        if homonym not in homonym_glosses:
            homonym_glosses[homonym] = set()
        homonym_glosses[homonym].add(gloss)
    
    # Second pass: create instances
    instances = []
    for id, item in data.items():
        inst = WSDInstance(
            id=str(id),
            homonym=item['homonym'],
            judged_meaning=item['judged_meaning'],
            precontext=item.get('precontext', ''),
            sentence=item['sentence'],
            ending=item.get('ending', ''),
            example_sentence=item.get('example_sentence', ''),
            average_score=float(item.get('average', 3.0)),
            all_glosses=list(homonym_glosses[item['homonym']])
        )
        instances.append(inst)
    
    return instances


def main():
    # Configuration
    MODEL_NAME = "microsoft/deberta-v3-base"
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 2e-5
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")
    
    # Load data
    print("Loading data...")
    train_instances = load_data('data/train.json')
    dev_instances = load_data('data/dev.json')
    
    print(f"Train instances: {len(train_instances)}")
    print(f"Dev instances: {len(dev_instances)}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = WSDDataset(train_instances, tokenizer)
    dev_dataset = WSDDataset(dev_instances, tokenizer)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    print("Initializing model...")
    model = MultiTaskWSDModel(model_name=MODEL_NAME).to(DEVICE)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_mae = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_mse, train_contrastive = train_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, epoch
        )
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train MSE: {train_mse:.4f}")
        print(f"Train Contrastive: {train_contrastive:.4f}")
        
        # Evaluate
        dev_results = evaluate(model, dev_loader, DEVICE)
        
        print(f"\nDev Results:")
        print(f"  MSE: {dev_results['mse']:.4f}")
        print(f"  MAE: {dev_results['mae']:.4f}")
        print(f"  Pearson: {dev_results['pearson']:.4f}")
        print(f"  Spearman: {dev_results['spearman']:.4f}")
        
        print(f"\nTask-specific MAE:")
        for task_name, metrics in dev_results['task_metrics'].items():
            print(f"  {task_name}: {metrics['mae']:.4f}")
        
        # Print learned task weights
        weights = model.task_weights.detach().cpu()
        weights = F.softmax(weights, dim=0).numpy()
        print(f"\nLearned Task Weights:")
        print(f"  ConSeC: {weights[0]:.4f}")
        print(f"  Narrative: {weights[1]:.4f}")
        print(f"  Contrastive: {weights[2]:.4f}")
        print(f"  Similarity: {weights[3]:.4f}")
        
        # Save best model
        if dev_results['mae'] < best_mae:
            best_mae = dev_results['mae']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mae': best_mae,
            }, 'best_multitask_wsd_model.pt')
            print(f"\n✓ Best model saved (MAE: {best_mae:.4f})")
    
    # Load best model and generate predictions
    print("\n" + "="*60)
    print("Generating final predictions...")
    print("="*60)
    
    checkpoint = torch.load('best_multitask_wsd_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    dev_results = evaluate(model, dev_loader, DEVICE)
    
    # Save predictions
    os.makedirs('predictions', exist_ok=True)
    
    with open('predictions/multitask_wsd_predictions.jsonl', 'w') as f:
        for id, pred in zip(dev_results['ids'], dev_results['predictions']):
            f.write(json.dumps({
                'id': id,
                'prediction': int(round(pred))
            }) + '\n')
    
    print(f"\nFinal Results:")
    print(f"  MSE: {dev_results['mse']:.4f}")
    print(f"  MAE: {dev_results['mae']:.4f}")
    print(f"  Pearson: {dev_results['pearson']:.4f}")
    print(f"  Spearman: {dev_results['spearman']:.4f}")
    
    print("\nPredictions saved to predictions/multitask_wsd_predictions.jsonl")


if __name__ == "__main__":
    main()