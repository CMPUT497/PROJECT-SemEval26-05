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

class RelevanceMLP(nn.Module):
    """A simple regression head that learns the optimal weights for the relevance subfeatures and penalty/scale."""
    def __init__(self):
        super().__init__()
        # Input: [relevance, coherence, confidence, wn_sim, penalty]
        # Output: single regression value (unscaled relevance 0-1)
        self.linear = nn.Linear(5, 1)
        # Learn a scale (to multiply with output for 1-5 range)
        self.scale = nn.Parameter(torch.tensor(3.75))

    def forward(self, features):
        """
        Args:
            features: [batch, 5]
        Returns:
            [batch, 1] (unscaled, to be scaled for 1-5 prediction)
        """
        out = self.linear(features)
        scaled = torch.sigmoid(out) * self.scale
        # Clamp to [1, 5] for safety
        final = torch.clamp(scaled, 1.0, 5.0)
        return final.squeeze(-1)  # [batch]

class DeBERTaWSDSystem:
    def __init__(self, model_name: str = "microsoft/deberta-v3-large", mlp_path: str = None, train_relevance_head: bool = False):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on: {self.device}")

        self.train_relevance_head = train_relevance_head
        self.relevance_head = RelevanceMLP()
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
        penalty_weight = 1.0
        if wsd_input.ending == "":
            penalty_weight = 0.8
        context_emb = self.get_contextual_embedding(full_context, wsd_input.homonym)
        example_emb = self.get_contextual_embedding(
            wsd_input.example_sentence, 
            wsd_input.homonym
        )
        gloss_emb = self.get_gloss_embedding(wsd_input.homonym, wsd_input.judged_meaning)
        context_example_sim = self.compute_similarity(context_emb, example_emb)
        context_gloss_sim = self.compute_similarity(context_emb, gloss_emb)
        combined_sim = penalty_weight * (0.3 * context_example_sim + 
                       0.7 * context_gloss_sim )
        sense_match = combined_sim >= threshold
        
        details = {
            'context_example_similarity': float(context_example_sim),
            'context_gloss_similarity': float(context_gloss_sim),
            'combined_similarity': float(combined_sim),
            'threshold': threshold
        }
        
        return sense_match, details

    def extract_relevance_features(self, wsd_input: WSDInput) -> np.ndarray:
        # This is the "feature" vector for regression
        full_context = ' '.join(wsd_input.precontext) + ' ' + wsd_input.sentence + ' ' + wsd_input.ending
        context_emb = self.get_contextual_embedding(full_context, wsd_input.homonym)
        example_emb = self.get_contextual_embedding(
            wsd_input.example_sentence, 
            wsd_input.homonym
        )
        gloss_emb = self.get_gloss_embedding(wsd_input.homonym, wsd_input.judged_meaning)
        penalty_weight = 1.0 if wsd_input.ending != "" else 0.95
        relevance = self.compute_similarity(context_emb, gloss_emb)
        coherence = self.compute_similarity(context_emb, example_emb)
        example_gloss_sim = self.compute_similarity(example_emb, gloss_emb)
        wn_sim = self.get_wordnet_similarity(wsd_input.homonym, wsd_input.judged_meaning)
        similarities = [relevance, coherence, example_gloss_sim, wn_sim]
        variance = np.var(similarities)
        confidence_score = max(0, 1 - variance)
        # NOTE: Order: [relevance, coherence, confidence, wn_sim, penalty_weight]
        feats = np.array([
            float(relevance),
            float(coherence),
            float(confidence_score),
            float(wn_sim),
            float(penalty_weight)
        ], dtype=np.float32)
        return feats  # shape (5,)

    def compute_relevance_score(self, wsd_input: WSDInput) -> Tuple[float, Dict]:
        # Extract features
        feats = self.extract_relevance_features(wsd_input)  # [5]
        feats_tensor = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            self.relevance_head.eval()
            result = self.relevance_head(feats_tensor)
            relevance_score = result.item()
        # For debugging, also include raw features
        details = {
            'input_features': feats.tolist(),
            'model_output': float(relevance_score)
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

def process_dataset(wsd_system: DeBERTaWSDSystem, data: Dict) -> List[Dict]:
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

def collect_training_examples(wsd_system: DeBERTaWSDSystem, data: Dict):
    """Extracts feature matrix X and target y from dataset using provided DeBERTa model."""
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

def train_relevance_head(wsd_system: DeBERTaWSDSystem, train_data: Dict, save_path: str = "relevance_head.pt", epochs: int = 60, lr: float = 1e-2):
    X_feats, y_targets = collect_training_examples(wsd_system, train_data)
    device = wsd_system.device
    mlp = wsd_system.relevance_head
    mlp.train()
    X_tensor = torch.tensor(X_feats, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_targets, dtype=torch.float32, device=device)
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    N = len(X_feats)
    best_loss = float("inf")
    for epoch in range(epochs):
        mlp.train()
        optimizer.zero_grad()
        preds = mlp(X_tensor)
        loss = loss_fn(preds, y_tensor)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            l = loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {l:.4f}")
        if l < best_loss:
            best_loss = l
            torch.save(mlp.state_dict(), save_path)
    print(f"Best loss: {best_loss:.4f}. Model saved to {save_path}")

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

def run(train_file, mlp_weights_path=None):
    data, _ = load_json_data(train_file)
    wsd_system = DeBERTaWSDSystem(mlp_path=mlp_weights_path)
    results = process_dataset(wsd_system, data)
    return results

def train_main():
    train_file = 'data/train.json'
    # 1. Load training data
    train_data, _ = load_json_data(train_file)
    # 2. Build DeBERTa model system with untrained RelevanceMLP
    wsd_system = DeBERTaWSDSystem()
    # 3. Train relevance head
    train_relevance_head(wsd_system, train_data, save_path="relevance_head.pt", epochs=60, lr=1e-2)

def inference_main():
    dev_file = 'data/dev.json'
    dev_data, file_dict = load_json_data(dev_file)
    # Load trained mlp weights for inference
    mlp_weights_path = "relevance_head.pt"
    wsd_system = DeBERTaWSDSystem(mlp_path=mlp_weights_path)
    # Run inference
    results = process_dataset(wsd_system, dev_data)
    predictions = [result['predicted_relevance_score'] for result in results]
    with open(f"predictions/wsd_predictions.JSONL", "w", encoding='utf8') as outfile:
        idx = 0
        for id in file_dict.keys():
            entry = {"id": id, "prediction": int(predictions[idx])}
            idx += 1
            outfile.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["train", "infer"], default="infer",
        help="Choose 'train' to train the weights, or 'infer' to run prediction (after training)."
    )
    args = parser.parse_args()
    if args.mode == "train":
        train_main()
    else:
        inference_main()
