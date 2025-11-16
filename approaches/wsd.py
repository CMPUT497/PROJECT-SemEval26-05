import os
import csv
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Tuple, List
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import nltk
from scipy.spatial.distance import cosine
from dataclasses import dataclass
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

@dataclass
class WSDInput:
    homonym: str
    judged_meaning: str  # gloss/definition
    precontext: List[str]  # 3 sentences
    sentence: str  # contains homonym
    ending: str  # 1 sentence
    example_sentence: str  # contains homonym

@dataclass
class WSDOutput:
    sanity_check_passed: bool
    sense_match: bool
    relevance_score: float  # 1-5
    details: Dict


class ImprovedRelevanceMLP(nn.Module):
    """
    Enhanced MLP for learning plausibility scores with better expressiveness
    and direct output in [1, 5] range without post-hoc scaling.
    Now uses 7 features including ending-specific metrics.
    """
    def __init__(self, input_dim=7, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        # Input normalization layer (learnable)
        self.input_norm = nn.BatchNorm1d(input_dim, affine=True)
        
        # Build deeper network with residual connections
        layers = []
        prev_dim = input_dim
        
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))  # More stable than BatchNorm for small batches
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = h
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer: 5 neurons for ordinal regression (scores 1-5)
        self.output_layer = nn.Linear(prev_dim, 5)
        
        # Initialize weights for better gradient flow
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier/Kaiming initialization for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: [batch, 7]
        Returns:
            [batch] scores in range [1, 5]
        """
        # Normalize inputs
        x = self.input_norm(features)
        
        # Pass through hidden layers
        x = self.hidden_layers(x)
        
        # Output layer gives logits for 5 classes
        logits = self.output_layer(x)  # [batch, 5]
        
        # Use softmax to get probabilities, then compute expected value
        probs = torch.softmax(logits, dim=-1)  # [batch, 5]
        
        # Expected value: sum of (class * probability)
        # Classes are 1, 2, 3, 4, 5
        class_values = torch.arange(1, 6, dtype=torch.float32, device=features.device)
        scores = torch.sum(probs * class_values, dim=-1)  # [batch]
        
        return scores


class FocalMSELoss(nn.Module):
    """
    Focal loss variant for MSE to focus on hard examples
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        mse = (predictions - targets) ** 2
        # Weight larger errors more heavily
        weights = (1 + mse) ** self.gamma
        return (weights * mse).mean()


class BERTWSDSystem:
    def __init__(self, model_name: str = "bert-large-uncased-whole-word-masking", 
                 mlp_path: str = None, 
                 train_relevance_head: bool = False, 
                 model_type: str = "softmax"):

        print(f"Loading BERT model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on: {self.device}")

        self.train_relevance_head = train_relevance_head
        self.model_type = model_type
        
        # Always use ImprovedRelevanceMLP with 7 features
        self.relevance_head = ImprovedRelevanceMLP(input_dim=7, hidden_dims=[512, 256, 128])
        
        if mlp_path is not None and os.path.exists(mlp_path):
            print(f"Loading trained relevance head from {mlp_path}")
            self.relevance_head.load_state_dict(torch.load(mlp_path, map_location=self.device))
        self.relevance_head.to(self.device)
        self.relevance_head.eval()

    @staticmethod
    def _l2_norm(x: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(x) + 1e-12
        return x / denom

    @staticmethod
    def _torch_l2_norm(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return t / (t.norm(dim=-1, keepdim=True) + eps)
    
    def _find_token_indices_for_target(self, sentence: str, target: str, encodings) -> List[int]:
        lower_sentence = sentence.lower()
        lower_target = target.lower()
        start_char = lower_sentence.find(lower_target)
        if start_char == -1:
            words = sentence.split()
            for i, w in enumerate(words):
                if lower_target in w.lower():
                    pos = 0
                    for j in range(i):
                        pos += len(words[j]) + 1
                    start_char = pos
                    break
        if start_char == -1:
            return []
        end_char = start_char + len(lower_target)
        offsets = encodings["offset_mapping"][0].tolist()
        token_indices = []
        for idx, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if not (e <= start_char or s >= end_char):
                token_indices.append(idx)
        return token_indices
    
    def get_contextual_embedding(self, sentence: str, target_word: str, layer_pool: str = "mean_last4") -> np.ndarray:
        enc = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            padding=False
        )
        enc = {k: v.to(self.device) for k, v in enc.items() if k != "offset_mapping"}
        raw_enc = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            padding=False
        )
        offset_mapping = raw_enc["offset_mapping"]
        with torch.no_grad():
            outputs = self.model(**enc, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        enc_with_offsets = {"offset_mapping": offset_mapping}
        token_idxs = self._find_token_indices_for_target(sentence, target_word, enc_with_offsets)
        if not token_idxs:
            last = hidden_states[-1][0]
            try:
                offsets = offset_mapping[0].tolist()
                real_token_mask = [not (s == 0 and e == 0) for (s, e) in offsets]
                real_idxs = [i for i, ok in enumerate(real_token_mask) if ok]
                vec = last[real_idxs].mean(dim=0).cpu().numpy()
            except Exception:
                vec = last.mean(dim=0).cpu().numpy()
            return self._l2_norm(vec)
        if layer_pool == "last":
            layer_vecs = hidden_states[-1][0]
        elif layer_pool == "mean_last4":
            last_k = torch.stack([hidden_states[-i][0] for i in range(1, 5)], dim=0)
            layer_vecs = last_k.mean(dim=0)
        elif layer_pool == "mean_all":
            all_layers = torch.stack([h[0] for h in hidden_states], dim=0)
            layer_vecs = all_layers.mean(dim=0)
        else:
            raise ValueError("Unknown layer_pool")
        target_tensor = layer_vecs[token_idxs]
        emb = target_tensor.mean(dim=0).cpu().numpy()
        return self._l2_norm(emb)
    
    def get_gloss_embedding(self, target_word: str, gloss: str, layer_pool: str = "mean_last4") -> np.ndarray:
        gloss_sentence = f"The word {target_word} means {gloss}"
        enc = self.tokenizer(
            gloss_sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            padding=False
        )
        enc_model = {k: v.to(self.device) for k, v in enc.items() if k != "offset_mapping"}
        offset_mapping = enc["offset_mapping"]
        with torch.no_grad():
            outputs = self.model(**enc_model, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        if layer_pool == "last":
            layer_vecs = hidden_states[-1][0]
        elif layer_pool == "mean_last4":
            last_k = torch.stack([hidden_states[-i][0] for i in range(1, 5)], dim=0)
            layer_vecs = last_k.mean(dim=0)
        elif layer_pool == "mean_all":
            all_layers = torch.stack([h[0] for h in hidden_states], dim=0)
            layer_vecs = all_layers.mean(dim=0)
        else:
            raise ValueError("Unknown layer_pool")
        try:
            offsets = offset_mapping[0].tolist()
            real_token_mask = [not (s == 0 and e == 0) for (s, e) in offsets]
            real_idxs = [i for i, ok in enumerate(real_token_mask) if ok]
            gloss_mean = layer_vecs[real_idxs].mean(dim=0)
        except Exception:
            gloss_mean = layer_vecs.mean(dim=0)
        token_idxs = self._find_token_indices_for_target(gloss_sentence, target_word, {"offset_mapping": offset_mapping})
        if token_idxs:
            target_emb = layer_vecs[token_idxs].mean(dim=0)
            combined = 0.3 * target_emb + 0.7 * gloss_mean
        else:
            combined = gloss_mean
        emb = combined.cpu().numpy()
        return self._l2_norm(emb)
    
    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        a = self._l2_norm(vec_a)
        b = self._l2_norm(vec_b)
        return float(np.dot(a, b))
    
    def get_wordnet_similarity(self, word: str, gloss: str) -> float:
        synsets = wn.synsets(word)
        if not synsets:
            return 0.0
        match = 0.0
        gloss_lower = gloss.lower()
        for synset in synsets:
            syn_gloss = synset.definition().lower()
            if gloss_lower == syn_gloss:
                match = 1.0
                break        
        return match
    
    def sanity_check(self, wsd_input: WSDInput) -> Tuple[bool, Dict]:
        example_emb = self.get_contextual_embedding(
            wsd_input.example_sentence, 
            wsd_input.homonym
        )
        gloss_emb = self.get_gloss_embedding(wsd_input.homonym, wsd_input.judged_meaning)
        similarity = self.compute_similarity(example_emb, gloss_emb)
        wn_similarity = self.get_wordnet_similarity(
            wsd_input.homonym, 
            wsd_input.judged_meaning
        )
        combined_score = 0.3 * similarity + 0.7 * wn_similarity
        passed = combined_score > 0.8
        details = {
            'embedding_similarity': float(similarity),
            'wordnet_similarity': float(wn_similarity),
            'combined_score': float(combined_score),
            'threshold': 0.8
        }
        return passed, details
    
    def compute_sense_match(self, wsd_input: WSDInput, 
                           threshold: float = 0.8) -> Tuple[bool, Dict]:
        full_context = ' '.join(wsd_input.precontext) + ' ' + wsd_input.sentence + ' ' + wsd_input.ending
        
        # No penalty - we now use ending information properly
        context_emb = self.get_contextual_embedding(full_context, wsd_input.homonym)
        example_emb = self.get_contextual_embedding(
            wsd_input.example_sentence, 
            wsd_input.homonym
        )
        gloss_emb = self.get_gloss_embedding(wsd_input.homonym, wsd_input.judged_meaning)
        context_example_sim = self.compute_similarity(context_emb, example_emb)
        context_gloss_sim = self.compute_similarity(context_emb, gloss_emb)
        combined_sim = 0.3 * context_example_sim + 0.7 * context_gloss_sim
        sense_match = combined_sim >= threshold
        
        details = {
            'context_example_similarity': float(context_example_sim),
            'context_gloss_similarity': float(context_gloss_sim),
            'combined_similarity': float(combined_sim),
            'threshold': threshold
        }
        
        return sense_match, details

    def extract_relevance_features(self, wsd_input: WSDInput) -> np.ndarray:
        """
        Extract features for relevance scoring with explicit ending integration
        Now returns 7 features instead of 5:
        [relevance, coherence, confidence, wn_sim, ending_coherence, ending_gloss_sim, has_ending]
        """
        full_context = ' '.join(wsd_input.precontext) + ' ' + wsd_input.sentence + ' ' + wsd_input.ending
        context_emb = self.get_contextual_embedding(full_context, wsd_input.homonym)
        example_emb = self.get_contextual_embedding(
            wsd_input.example_sentence, 
            wsd_input.homonym
        )
        gloss_emb = self.get_gloss_embedding(wsd_input.homonym, wsd_input.judged_meaning)
        
        # Core features
        relevance = self.compute_similarity(context_emb, gloss_emb)
        coherence = self.compute_similarity(context_emb, example_emb)
        example_gloss_sim = self.compute_similarity(example_emb, gloss_emb)
        wn_sim = self.get_wordnet_similarity(wsd_input.homonym, wsd_input.judged_meaning)
        
        # NEW: Ending-specific features
        has_ending = 1.0 if wsd_input.ending.strip() else 0.0
        
        if has_ending:
            # Get embedding for ending sentence specifically
            ending_emb = self.get_contextual_embedding(wsd_input.ending, wsd_input.homonym)
            
            # How well does ending cohere with the main context?
            context_without_ending = ' '.join(wsd_input.precontext) + ' ' + wsd_input.sentence
            context_no_ending_emb = self.get_contextual_embedding(context_without_ending, wsd_input.homonym)
            ending_coherence = self.compute_similarity(ending_emb, context_no_ending_emb)
            
            # How well does ending match the intended sense?
            ending_gloss_sim = self.compute_similarity(ending_emb, gloss_emb)
        else:
            ending_coherence = 0.0
            ending_gloss_sim = 0.0
        
        similarities = [relevance, coherence, example_gloss_sim, wn_sim]
        variance = np.var(similarities)
        confidence_score = max(0, 1 - variance)
        
        # Feature vector: [relevance, coherence, confidence, wn_sim, ending_coherence, ending_gloss_sim, has_ending]
        feats = np.array([
            float(relevance),
            float(coherence),
            float(confidence_score),
            float(wn_sim),
            float(ending_coherence),
            float(ending_gloss_sim),
            float(has_ending)
        ], dtype=np.float32)
        return feats  # shape (7,)

    def compute_relevance_score(self, wsd_input: WSDInput) -> Tuple[float, Dict]:
        # Extract features (now 7 features)
        feats = self.extract_relevance_features(wsd_input)  # [7]
        feats_tensor = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            self.relevance_head.eval()
            result = self.relevance_head(feats_tensor)
            relevance_score = result.item()
        # For debugging, also include raw features
        details = {
            'input_features': feats.tolist(),
            'model_output': float(relevance_score),
            'feature_names': ['relevance', 'coherence', 'confidence', 'wn_sim', 
                            'ending_coherence', 'ending_gloss_sim', 'has_ending']
        }
        return relevance_score, details

    def process(self, wsd_input: WSDInput, 
                similarity_threshold: float = 0.8) -> WSDOutput:
        sanity_passed, sanity_details = self.sanity_check(wsd_input)
        sense_match, match_details = self.compute_sense_match(
            wsd_input, 
            threshold=similarity_threshold
        )
        relevance_score, relevance_details = self.compute_relevance_score(wsd_input)
        all_details = {
            'sanity_check': sanity_details,
            'sense_matching': match_details,
            'relevance_scoring': relevance_details
        }
        return WSDOutput(
            sanity_check_passed=sanity_passed,
            sense_match=sense_match,
            relevance_score=float(relevance_score),
            details=all_details
        )


def process_dataset(wsd_system: BERTWSDSystem, data: Dict) -> List[Dict]:
    results = []
    num_examples = len(data['homonym_word'])
    for i in range(num_examples):
        context_text = data['context_sentences'][i]
        precontext = [s.strip() + '.' for s in context_text.split('.') if s.strip()]
        wsd_input = WSDInput(
            homonym=data['homonym_word'][i],
            judged_meaning=data['judged_meaning'][i],
            precontext=precontext,
            sentence=data['ambiguous_sentence'][i],
            ending=data['ending_sentence'][i],
            example_sentence=data['example_sentence'][i]
        )
        result = wsd_system.process(wsd_input, similarity_threshold=0.8)
        result_dict = {
            'index': i,
            'homonym': data['homonym_word'][i],
            'judged_meaning': data['judged_meaning'][i],
            'sanity_check_passed': result.sanity_check_passed,
            'sense_match': result.sense_match,
            'predicted_relevance_score': round(result.relevance_score, 2),
            'ground_truth_score': data.get('score', [None] * num_examples)[i],
            'details': result.details
        }
        results.append(result_dict)
    return results


def collect_training_examples(wsd_system: BERTWSDSystem, data: Dict):
    """Extracts feature matrix X and target y from dataset using provided SenseBERT model."""
    X_feats = []
    y = []
    num_examples = len(data['homonym_word'])
    for i in range(num_examples):
        context_text = data['context_sentences'][i]
        precontext = [s.strip() + '.' for s in context_text.split('.') if s.strip()]
        wsd_input = WSDInput(
            homonym=data['homonym_word'][i],
            judged_meaning=data['judged_meaning'][i],
            precontext=precontext,
            sentence=data['ambiguous_sentence'][i],
            ending=data['ending_sentence'][i],
            example_sentence=data['example_sentence'][i]
        )
        feats = wsd_system.extract_relevance_features(wsd_input)
        X_feats.append(feats)
        score = data['score'][i] if 'score' in data else None
        y.append(float(score))
    X_feats = np.stack(X_feats, axis=0)
    y = np.array(y, dtype=np.float32)
    return X_feats, y


def train_relevance_head_improved(
    wsd_system,
    train_data: Dict,
    save_path: str = "relevance_head_improved.pt",
    epochs: int = 300,
    lr: float = 5e-3
):
    """
    Improved training procedure with better optimization and regularization
    """
    # Extract features
    X_feats, y_targets = collect_training_examples(wsd_system, train_data)
    device = wsd_system.device
    
    # Use the model already created in wsd_system
    mlp = wsd_system.relevance_head
    mlp.to(device)
    mlp.train()
    
    # Prepare data
    X_tensor = torch.tensor(X_feats, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_targets, dtype=torch.float32, device=device)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Loss function: Use Huber for robustness
    loss_fn = nn.HuberLoss(delta=1.0)
    
    best_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in range(epochs):
        mlp.train()
        optimizer.zero_grad()
        
        preds = mlp(X_tensor)
        loss = loss_fn(preds, y_tensor)
        
        # Add variance penalty to encourage spread in predictions
        pred_var = preds.var()
        target_var = y_tensor.var()
        variance_loss = (pred_var - target_var).abs()
        
        # Combined loss
        total_loss = loss + 0.1 * variance_loss
        
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        with torch.no_grad():
            l = total_loss.item()
            pred_std = preds.std().item()
            target_std = y_tensor.std().item()
        
        # Update learning rate
        scheduler.step(l)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {l:.4f} - "
                  f"Pred std: {pred_std:.3f} - Target std: {target_std:.3f}")
            
            # Show distribution of predictions
            with torch.no_grad():
                pred_min = preds.min().item()
                pred_max = preds.max().item()
                pred_mean = preds.mean().item()
                print(f"  Predictions: min={pred_min:.2f}, max={pred_max:.2f}, mean={pred_mean:.2f}")
        
        # Early stopping
        if l < best_loss:
            best_loss = l
            patience_counter = 0
            torch.save(mlp.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Best loss: {best_loss:.4f}. Model saved to {save_path}")
    
    # Load best model and return final predictions
    mlp.load_state_dict(torch.load(save_path))
    mlp.eval()
    with torch.no_grad():
        final_preds = mlp(X_tensor)
        print("\n=== Final Training Set Predictions ===")
        print(f"Predictions - Min: {final_preds.min():.2f}, Max: {final_preds.max():.2f}, "
              f"Mean: {final_preds.mean():.2f}, Std: {final_preds.std():.2f}")
        print(f"Targets - Min: {y_tensor.min():.2f}, Max: {y_tensor.max():.2f}, "
              f"Mean: {y_tensor.mean():.2f}, Std: {y_tensor.std():.2f}")


def load_json_data(json_file: str):
    with open(json_file, 'r', encoding='utf8') as f:
        file_dict = json.load(f)
    ambigous_sentences, homonym_words, context_sentences, judged_meanings, ending_sentences, example_sentences, scores = [], [], [], [], [], [], []
    for item in file_dict.values():
        homonym_words.append(item['homonym'])
        context_sentences.append(item['precontext'])
        judged_meanings.append(item['judged_meaning'])
        ending_sentences.append(item['ending'])
        example_sentences.append(item['example_sentence'])
        ambigous_sentences.append(item['sentence'])
        scores.append(item['average'])
    data = {
        'homonym_word': homonym_words,
        'context_sentences': context_sentences,
        'ambiguous_sentence': ambigous_sentences,
        'judged_meaning': judged_meanings,
        'ending_sentence': ending_sentences,
        'example_sentence': example_sentences,
        'score': scores
    }

    return data, file_dict


def run(train_file, mlp_weights_path=None, model_type="softmax"):
    data, _ = load_json_data(train_file)
    wsd_system = BERTWSDSystem(mlp_path=mlp_weights_path, model_type=model_type)
    results = process_dataset(wsd_system, data)
    return results


def train_main(model_type="softmax"):
    """
    Train the relevance head using improved architecture with SenseBERT
    """
    train_file = 'data/train.json'
    # 1. Load training data
    train_data, _ = load_json_data(train_file)
    # 2. Build SenseBERT model system with untrained RelevanceMLP
    wsd_system = BERTWSDSystem(model_type="softmax")
    # 3. Train relevance head with improved training procedure
    save_path = f"relevance_head_sensebert_softmax.pt"
    train_relevance_head_improved(
        wsd_system, 
        train_data, 
        save_path=save_path, 
        epochs=300, 
        lr=5e-3
    )
    print(f"\nTraining complete! Model saved to {save_path}")


def inference_main(model_type="softmax"):
    """
    Run inference using trained model with SenseBERT
    """
    dev_file = 'data/dev.json'
    dev_data, file_dict = load_json_data(dev_file)
    # Load trained mlp weights for inference
    mlp_weights_path = f"relevance_head_sensebert_softmax.pt"
    
    if not os.path.exists(mlp_weights_path):
        print(f"Error: Model file '{mlp_weights_path}' not found!")
        print(f"Please run training first with --mode train")
        return
    
    wsd_system = BERTWSDSystem(mlp_path=mlp_weights_path, model_type="softmax")
    # Run inference
    results = process_dataset(wsd_system, dev_data)
    predictions = [result['predicted_relevance_score'] for result in results]
    
    # Create predictions directory if it doesn't exist
    os.makedirs("predictions", exist_ok=True)
    
    output_file = f"predictions/wsd_predictions_sensebert_softmax.JSONL"
    with open(output_file, "w", encoding='utf8') as outfile:
        idx = 0
        for id in file_dict.keys():
            entry = {"id": id, "prediction": int(predictions[idx])}
            idx += 1
            outfile.write(json.dumps(entry) + "\n")
    
    print(f"\nInference complete! Predictions saved to {output_file}")
    print(f"\nPrediction Statistics:")
    print(f"  Min: {min(predictions):.2f}")
    print(f"  Max: {max(predictions):.2f}")
    print(f"  Mean: {np.mean(predictions):.2f}")
    print(f"  Std: {np.std(predictions):.2f}")
    
    # Show distribution
    from collections import Counter
    rounded_preds = [round(p) for p in predictions]
    dist = Counter(rounded_preds)
    print(f"\nPrediction Distribution:")
    for score in sorted(dist.keys()):
        print(f"  Score {score}: {dist[score]} samples ({100*dist[score]/len(predictions):.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='WSD System with SenseBERT and Improved Relevance Scoring')
    parser.add_argument(
        "--mode", 
        choices=["train", "infer"], 
        default="infer",
        help="Choose 'train' to train the weights, or 'infer' to run prediction (after training)."
    )
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Running SenseBERT WSD System in {args.mode.upper()} mode")
    print(f"{'='*60}\n")
    
    if args.mode == "train":
        train_main()
    else:
        inference_main()
