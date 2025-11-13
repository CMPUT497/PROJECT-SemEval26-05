import os
import csv
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.pipelines import pipeline
from scipy.spatial.distance import cosine
from typing import Dict, Tuple, List
from dataclasses import dataclass
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class WSDInput:
    homonym: str
    judged_meaning: str  
    precontext: List[str] 
    sentence: str  
    ending: str  
    example_sentence: str  

@dataclass
class WSDOutput:
    sanity_check_passed: bool
    sense_match: bool
    relevance_score: float 
    details: Dict

class UrduWSDSystem:
    def __init__(self, 
                 nllb_model='facebook/nllb-200-distilled-600M', 
                 embedder_model='sentence-transformers/LaBSE'):
        

        self.translator = pipeline(
            'translation',
            model=nllb_model,
            src_lang='eng_Latn',
            tgt_lang='urd_Arab',
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.embedder_tokenizer = AutoTokenizer.from_pretrained(embedder_model)
        self.embedder_model = AutoModel.from_pretrained(embedder_model)
        self.embedder_model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder_model.to(self.device)

    @staticmethod
    def _l2_norm(x: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(x) + 1e-12
        return x / denom

    def _get_embedding(self, sentence: str) -> np.ndarray:
        """Generates a LaBSE (multilingual) embedding for a given sentence."""
        inputs = self.embedder_tokenizer(
            sentence, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.embedder_model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        
        return self._l2_norm(embeddings)

    def _translate(self, texts: List[str]) -> List[str]:
        """Translates a list of English texts to Urdu using NLLB."""
        try:
            translations = self.translator(texts)
            return [t['translation_text'] for t in translations]
        except Exception as e:
            print(f"Translation Error: {e}")
            return [""] * len(texts)

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Returns cosine similarity between two L2-normalized vectors."""
        return float(1 - cosine(vec_a, vec_b))

    
    def calculate_similarity_scores(self, wsd_input: WSDInput) -> Dict:
        """
        Calculates the two required similarity scores (Context vs Example and Context vs Gloss)
        in the multilingual embedding space.
        """
        full_context_raw = " ".join(wsd_input.precontext) + " " + wsd_input.sentence + " " + wsd_input.ending        

        inputs_to_translate = [
            full_context_raw,
            wsd_input.example_sentence,
            wsd_input.judged_meaning 
        ]
        
        clean_inputs = [' '.join(text.split()) for text in inputs_to_translate]
        
        # Translate to Urdu
        urdu_context, urdu_example, urdu_gloss = self._translate(clean_inputs)

        if not urdu_context or not urdu_example or not urdu_gloss:
            return {
                'error': 'Translation failed', 
                'S_Context_vs_Example': 0.0,
                'S_Context_vs_Gloss': 0.0
            }

        # Embed the Urdu Translations
        vector_context = self._get_embedding(urdu_context)
        vector_example = self._get_embedding(urdu_example)
        vector_gloss = self._get_embedding(urdu_gloss)

        # Compute both similarity scores
        Context_vs_Gloss = self.compute_similarity(vector_context, vector_gloss) 
        Context_vs_Example = self.compute_similarity(vector_context, vector_example) 
        
        details = {
            'urdu_context_trans': urdu_context,
            'urdu_example_trans': urdu_example,
            'urdu_gloss_trans': urdu_gloss,
            'Context_vs_Gloss': float(Context_vs_Gloss),
            'Context_vs_Example': float(Context_vs_Example),
        }
        
        return details
    
    def compute_relevance_score(self, S1: float, S2: float, weight_S1: float = 0.5) -> Tuple[float, Dict]:
        """
        Computes the Relevance Score (1-5) based on the weighted average of two similarities.
        S1: Context vs Gloss (Weight_S1 = 0.5)
        S2: Context vs Example (Weight_S2 = 0.5)
        """
        weight_S2 = 1.0 - weight_S1
        
        # Weighted average 
        weighted_avg_similarity = (S1 * weight_S1) + (S2 * weight_S2)
        
        final_score = 1 + (weighted_avg_similarity * 4) 

        details = {
            'Context_vs_Gloss_W': weight_S1,
            'Context_vs_Example_W': weight_S2,
            'weighted_avg_similarity': float(weighted_avg_similarity),
        }
        
        return final_score, details
        
    
    def process(self, wsd_input: WSDInput, 
                similarity_threshold: float = 0.8) -> WSDOutput:
        
        all_match_details = self.calculate_similarity_scores(wsd_input)
        
        Context_vs_Gloss = all_match_details.get('Context_vs_Gloss', 0.0)
        Context_vs_Example = all_match_details.get('Context_vs_Example', 0.0)
        
        # Compute Relevance Score 
        relevance_score, relevance_details = self.compute_relevance_score(
            S1=Context_vs_Gloss, 
            S2=Context_vs_Example, 
            weight_S1=0.5
        )
        
        # sense_match 
        final_similarity = relevance_details['weighted_avg_similarity']
        sense_match = final_similarity >= similarity_threshold

        return WSDOutput(
            sanity_check_passed=True, 
            sense_match=sense_match,
            relevance_score=relevance_score,
            details={
                'similarity_scores': all_match_details,
                'relevance_scoring': relevance_details
            }
        )
    
def process_dataset(wsd_system: UrduWSDSystem, data: Dict) -> List[Dict]:
    """Processes the entire dataset using the UrduWSDSystem."""
    results = []
    
    num_examples = len(data['homonym_word'])
    
    for i in range(num_examples):
        context_text = data['context_sentences'][i]
        
        # Create WSD input
        wsd_input = WSDInput(
            homonym=data['homonym_word'][i],
            judged_meaning=data['judged_meaning'][i],
            precontext=[context_text], 
            sentence=data['ambiguous_sentence'][i],
            ending=data['ending_sentence'][i],
            example_sentence=data['example_sentence'][i]
        )
        
        # Process
        result = wsd_system.process(wsd_input, similarity_threshold=0.49)

        all_scores = result.details['similarity_scores']
        scoring_details = result.details['relevance_scoring']
        
        # Extract original English sentences (retaining the full context for manual inspection)
        full_context_raw = " ".join(wsd_input.precontext) + " " + wsd_input.sentence + " " + wsd_input.ending
        
        # Store results for TSV output
        result_dict = {
            'index': i,
            'homonym': data['homonym_word'][i],
            'judged_meaning': data['judged_meaning'][i],
            'sanity_check_passed': result.sanity_check_passed, 
            'sense_match': result.sense_match,
            'Context_vs_Gloss': all_scores['Context_vs_Gloss'], 
            'Context_vs_Example': all_scores['Context_vs_Example'],
            'predicted_relevance_score': round(result.relevance_score, 4),
            'ground_truth_score': data.get('score', [None] * num_examples)[i] if 'score' in data else None,
            'details': result.details
        }
        
        results.append(result_dict)
    
    return results

def run(file_path: str) -> List[Dict]:
    """Loads data, initializes the system, and runs the analysis."""
    print(f"Loading data from: {file_path}")
    
    with open(file_path, 'r', encoding='utf8') as f:
        file_dict = json.load(f)

    ambiguous_sentences, homonym_words, context_sentences, judged_meanings, ending_sentences, example_sentences = [], [], [], [], [], []
    scores = [] 

    for item in file_dict.values():
        homonym_words.append(item['homonym'])
        context_sentences.append(item['precontext'])
        judged_meanings.append(item['judged_meaning'])
        ending_sentences.append(item['ending'])
        example_sentences.append(item['example_sentence'])
        ambiguous_sentences.append(item['sentence'])
        scores.append(item.get('average'))
    
    data = {
        'homonym_word': homonym_words,
        'context_sentences': context_sentences,
        'ambiguous_sentence': ambiguous_sentences,
        'judged_meaning': judged_meanings,
        'ending_sentence': ending_sentences,
        'example_sentence': example_sentences,
        'score': scores
    }
    
    wsd_system = UrduWSDSystem()
    results = process_dataset(wsd_system, data)
    
    return results

def main():
    dev_file = 'data/dev.json' 
    
    results = run(dev_file)
    
    predictions = [result['predicted_relevance_score'] for result in results]
    with open(f"predictions/ohpt_predictions.JSONL", "w", encoding='utf8') as outfile:
        idx = 0
        for id in file_dict.keys():
            entry = {"id": id, "prediction": int(predictions[idx])}
            idx += 1
            outfile.write(json.dumps(entry) + "\n")
   

if __name__ == "__main__":
    main()
