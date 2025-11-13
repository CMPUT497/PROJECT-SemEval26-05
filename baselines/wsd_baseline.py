import os
import csv
import sys
import json
import torch
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


class DeBERTaWSDSystem:
    def __init__(self, model_name: str = "microsoft/deberta-v3-large"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on: {self.device}")
    
    @staticmethod
    def _l2_norm(x: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(x) + 1e-12
        return x / denom

    @staticmethod
    def _torch_l2_norm(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return t / (t.norm(dim=-1, keepdim=True) + eps)
    
    def _find_token_indices_for_target(self, sentence: str, target: str, encodings) -> List[int]:
        # find char span of target (first occurrence, case-insensitive)
        lower_sentence = sentence.lower()
        lower_target = target.lower()
        start_char = lower_sentence.find(lower_target)
        if start_char == -1:
            # fallback: try whitespace word matching
            words = sentence.split()
            for i, w in enumerate(words):
                if lower_target in w.lower():
                    # compute char start of this word
                    pos = 0
                    for j in range(i):
                        pos += len(words[j]) + 1
                    start_char = pos
                    break
        if start_char == -1:
            return []  # not found

        end_char = start_char + len(lower_target)

        # offset_mapping exists per token (list of (start, end) tuples in original string)
        offsets = encodings["offset_mapping"][0].tolist()  # list of tuples
        token_indices = []
        # note: offsets align to the raw sentence; special tokens may have (0,0)
        for idx, (s, e) in enumerate(offsets):
            # skip special tokens with (0,0) that do not correspond to text
            if s == 0 and e == 0:
                continue
            # if token span intersects target char span -> include
            if not (e <= start_char or s >= end_char):
                token_indices.append(idx)
        return token_indices
    
    def get_contextual_embedding(self, sentence: str, target_word: str, layer_pool: str = "mean_last4") -> np.ndarray:
        # Markers are not strictly necessary with offset mapping; avoid adding markers that break offsets.
        # Use fast tokenizer with offset mapping to find tokens covering the target.
        enc = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            padding=False
        )
        enc = {k: v.to(self.device) for k, v in enc.items() if k != "offset_mapping"}
        # Keep offset_mapping on cpu for lookup
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
            # outputs.hidden_states: tuple(layer_count, batch, seq_len, dim)
            hidden_states = outputs.hidden_states  # tuple

        # determine token indices for target using offsets
        enc_with_offsets = {"offset_mapping": offset_mapping}
        token_idxs = self._find_token_indices_for_target(sentence, target_word, enc_with_offsets)
        if not token_idxs:
            # fallback: mean pooling over the full sequence (excluding special tokens if possible)
            last = hidden_states[-1][0]  # [seq_len, dim]
            # attempt to exclude special tokens by using offset map
            try:
                offsets = offset_mapping[0].tolist()
                real_token_mask = [not (s == 0 and e == 0) for (s, e) in offsets]
                real_idxs = [i for i, ok in enumerate(real_token_mask) if ok]
                vec = last[real_idxs].mean(dim=0).cpu().numpy()
            except Exception:
                vec = last.mean(dim=0).cpu().numpy()
            return self._l2_norm(vec)

        # choose pooling of layers
        if layer_pool == "last":
            layer_vecs = hidden_states[-1][0]  # [seq_len, dim]
        elif layer_pool == "mean_last4":
            last_k = torch.stack([hidden_states[-i][0] for i in range(1, 5)], dim=0)  # [4, seq_len, dim]
            layer_vecs = last_k.mean(dim=0)  # [seq_len, dim]
        elif layer_pool == "mean_all":
            all_layers = torch.stack([h[0] for h in hidden_states], dim=0)  # [L, seq_len, dim]
            layer_vecs = all_layers.mean(dim=0)
        else:
            raise ValueError("Unknown layer_pool")

        # token indices returned use tokenizer indexing; we must be careful if offset mapping included special tokens
        # hidden_states indexing aligns with tokenizer output in enc (they correspond)
        target_tensor = layer_vecs[token_idxs]  # [num_target_tokens, dim]
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

        # overall mean of sentence (try to exclude special tokens via offsets)
        try:
            offsets = offset_mapping[0].tolist()
            real_token_mask = [not (s == 0 and e == 0) for (s, e) in offsets]
            real_idxs = [i for i, ok in enumerate(real_token_mask) if ok]
            gloss_mean = layer_vecs[real_idxs].mean(dim=0)
        except Exception:
            gloss_mean = layer_vecs.mean(dim=0)

        # try to find target token within gloss sentence to bias embedding slightly toward it
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
    
    # checks if the gloss(judged meaning) is present in the gloss of the homonym
    def get_wordnet_similarity(self, word: str, gloss: str) -> float:
        synsets = wn.synsets(word)
        if not synsets:
            return 0.0
        match = 0.0
        gloss_lower = gloss.lower()
        # print("Target word GLOSS:", gloss_lower)
        for synset in synsets:
            syn_gloss = synset.definition().lower()
            if gloss_lower == syn_gloss:
                # print("glosses from wordnet:", syn_gloss)
                match = 1.0
                break        
        return match
    
    # function to check if the judged meaning and the example sentence have the same sense
    def sanity_check(self, wsd_input: WSDInput) -> Tuple[bool, Dict]:
        # Get embedding for example sentence with homonym
        example_emb = self.get_contextual_embedding(
            wsd_input.example_sentence, 
            wsd_input.homonym
        )
        # Get embedding for judged meaning (gloss)
        gloss_emb = self.get_gloss_embedding(wsd_input.homonym, wsd_input.judged_meaning)
        # Compute similarity
        similarity = self.compute_similarity(example_emb, gloss_emb)
        # WordNet similarity
        wn_similarity = self.get_wordnet_similarity(
            wsd_input.homonym, 
            wsd_input.judged_meaning
        )
        # Combined score
        combined_score = 0.3 * similarity + 0.7 * wn_similarity
        # Threshold for sanity check (should be high)
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
        # Combine context
        full_context = ' '.join(wsd_input.precontext) + ' ' + wsd_input.sentence + ' ' + wsd_input.ending
        # penalising context without any ending
        penalty_weight = 1.0
        if wsd_input.ending == "":
            penalty_weight = 0.8
        # Get embeddings
        context_emb = self.get_contextual_embedding(full_context, wsd_input.homonym)
        example_emb = self.get_contextual_embedding(
            wsd_input.example_sentence, 
            wsd_input.homonym
        )
        gloss_emb = self.get_gloss_embedding(wsd_input.homonym, wsd_input.judged_meaning)
        # Compute similarities
        context_example_sim = self.compute_similarity(context_emb, example_emb)
        context_gloss_sim = self.compute_similarity(context_emb, gloss_emb)
        # Combined similarity (weighted average)
        combined_sim = penalty_weight * (0.3 * context_example_sim + 
                       0.7 * context_gloss_sim )
        # sense is marked as matching if it the similarity is above threshold
        sense_match = combined_sim >= threshold
        
        details = {
            'context_example_similarity': float(context_example_sim),
            'context_gloss_similarity': float(context_gloss_sim),
            'combined_similarity': float(combined_sim),
            'threshold': threshold
        }
        
        return sense_match, details
    
    def compute_relevance_score(self, wsd_input: WSDInput) -> Tuple[float, Dict]:
        # Combine context
        full_context = ' '.join(wsd_input.precontext) + ' ' + wsd_input.sentence + ' ' + wsd_input.ending
        # Get embeddings
        context_emb = self.get_contextual_embedding(full_context, wsd_input.homonym)
        example_emb = self.get_contextual_embedding(
            wsd_input.example_sentence, 
            wsd_input.homonym
        )
        gloss_emb = self.get_gloss_embedding(wsd_input.homonym, wsd_input.judged_meaning)
        # penalising context without any ending
        penalty_weight = 1.0
        if wsd_input.ending == "":
            penalty_weight = 0.95
        # 1. Relevance probability: how relevant is the context sense to target sense
        relevance = self.compute_similarity(context_emb, gloss_emb)
        relevance_prob = relevance
        # 2. Coherence score: how coherent is context sense with example sense
        coherence = self.compute_similarity(context_emb, example_emb)
        coherence_score = coherence
        # 3. Confidence score: consistency across all comparisons
        example_gloss_sim = self.compute_similarity(example_emb, gloss_emb)
        wn_sim = self.get_wordnet_similarity(wsd_input.homonym, wsd_input.judged_meaning)
        # Confidence based on variance (low variance = high confidence)
        similarities = [relevance, coherence, example_gloss_sim, wn_sim]
        # similarities = [relevance, coherence, example_gloss_sim]
        mean_sim = np.mean(similarities)
        variance = np.var(similarities)
        confidence_score = max(0, 1 - variance)  # Lower variance = higher confidence
        # Combined score (weighted average)
        combined = penalty_weight * (
                    0.35 * relevance_prob + 
                    0.35 * coherence_score + 
                    0.3 * confidence_score)
        # Scale to 1-5 range
        # final_score = 1 + (combined * 3.8)  # Maps [0,1] to [1,5]
        final_score = combined * 3.75 # Maps [0,1] to [1,5]
        # final_score = 1 + ((final_score - 3.0) * 4) # Maps [4,5] to [1,5]
              
        details = {
            'relevance_probability': float(relevance_prob),
            'coherence_score': float(coherence_score),
            'confidence_score': float(confidence_score),
            'combined_normalized': float(combined),
            'individual_similarities': {
                'context_gloss': float(relevance),
                'context_example': float(coherence),
                'example_gloss': float(example_gloss_sim),
                'wordnet': float(wn_sim)
            }
        }
        
        return final_score, details
    
    def process(self, wsd_input: WSDInput, 
                similarity_threshold: float = 0.8) -> WSDOutput:
        # Step 1: Sanity check
        sanity_passed, sanity_details = self.sanity_check(wsd_input)
        
        # Step 2: Sense matching
        sense_match, match_details = self.compute_sense_match(
            wsd_input, 
            threshold=similarity_threshold
        )
        
        # Step 3: Relevance score
        relevance_score, relevance_details = self.compute_relevance_score(wsd_input)
        
        # Combine all details
        all_details = {
            'sanity_check': sanity_details,
            'sense_matching': match_details,
            'relevance_scoring': relevance_details
        }
        
        return WSDOutput(
            sanity_check_passed=sanity_passed,
            sense_match=sense_match,
            relevance_score=relevance_score,
            details=all_details
        )

def process_dataset(wsd_system: DeBERTaWSDSystem, data: Dict) -> List[Dict]:
    results = []
    
    num_examples = len(data['homonym_word'])
    
    for i in range(num_examples):
        # Parse context sentences (split by periods)
        context_text = data['context_sentences'][i]
        precontext = [s.strip() + '.' for s in context_text.split('.') if s.strip()]
        
        # Create WSD input
        wsd_input = WSDInput(
            homonym=data['homonym_word'][i],
            judged_meaning=data['judged_meaning'][i],
            precontext=precontext,
            sentence=data['ambiguous_sentence'][i],
            ending=data['ending_sentence'][i],
            example_sentence=data['example_sentence'][i]
        )
        
        # Process
        result = wsd_system.process(wsd_input, similarity_threshold=0.8)
        
        # Store results
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

def run(train_file):
  
    with open(train_file, 'r', encoding='utf8') as f:
        file_dict = json.load(f)

    ambigous_sentences, homonym_words, context_sentences, judged_meanings, ending_sentences, example_sentences, scores = [], [], [], [], [], [], []
    
    for item in file_dict.values():
        homonym_words.append(item['homonym'])
        context_sentences.append(item['precontext'])
        judged_meanings.append(item['judged_meaning'])
        ending_sentences.append(item['ending'])
        example_sentences.append(item['example_sentence'])
        ambigous_sentences.append(item['sentence'])
    
    data = {
        'homonym_word': homonym_words,
        'context_sentences': context_sentences,
        'ambiguous_sentence': ambigous_sentences,
        'judged_meaning': judged_meanings,
        'ending_sentence': ending_sentences,
        'example_sentence': example_sentences,
    }
    
    # Initialize system
    wsd_system = DeBERTaWSDSystem()
    
    # Process all examples
    results = process_dataset(wsd_system, data)
    
    predictions = [result['predicted_relevance_score'] for i, result in enumerate(results, 1)]
    with open(f"predictions/wsd_baseline_predictions.JSONL", "a", encoding='utf8') as outfile:
        idx = 0
        for id in file_dict.keys():
            entry = {"id": id, "prediction": int(predictions[idx])}
            idx += 1
            outfile.write(json.dumps(entry) + "\n")


def main():
    dev_file = 'data/dev.json'
    run(dev_file)


if __name__ == "__main__":
    main()
