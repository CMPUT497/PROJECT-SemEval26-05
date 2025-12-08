import json
import os
import requests
import torch
import difflib
import numpy as np
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer, util

# FILE PATHS
INPUT_FILE = "../../data/dev.json"
OUTPUT_DIR = "../../predictions"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dev_predictions_wsd.jsonl")
CACHE_FILE = "processing_cache.json"

# API CONFIG
BABELNET_API_KEY = "9dce12de-cd8d-406c-a219-e8e2a77275cc" 
BABELNET_API_URL = "https://babelnet.io/v9/getSynsetIds"
BABELNET_INFO_URL = "https://babelnet.io/v9/getSynset"

# TRANSLATION MODEL
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# WSD EMBEDDING MODELS
EN_EMBEDDING_MODEL = "microsoft/deberta-base"
UR_EMBEDDING_MODEL = "xlm-roberta-base"

# DEVICE SETUP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 if torch.cuda.is_available() else -1
print(f"Running on: {DEVICE} (ID: {DEVICE_ID})")


class ResultCache:
    def __init__(self, filename=CACHE_FILE):
        self.filename = filename
        self.cache = {}
        self.load()

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    loaded_cache = json.load(f)
                
                valid_cache = {}
                cleaned_count = 0
                for k, v in loaded_cache.items():
                    if (v.get('gold_id') is not None and 
                        v.get('en_id') is not None):
                        valid_cache[k] = v
                    else:
                        cleaned_count += 1
                
                self.cache = valid_cache
                print(f"Loaded {len(self.cache)} valid items from cache. (Purged {cleaned_count} incomplete items)")
                
                if cleaned_count > 0:
                    self.save()
                    
            except json.JSONDecodeError:
                print("Cache file corrupted, starting fresh.")
                self.cache = {}

    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4)

    def get(self, sample_id):
        return self.cache.get(str(sample_id))

    def set(self, sample_id, data):
        if data.get('gold_id') and data.get('en_id'):
            self.cache[str(sample_id)] = data
            self.save()

result_cache = ResultCache()

# API HELPERS (BabelNet)
runtime_api_cache = {}

def fetch_babelnet_ids(lemma, lang="EN"):
    cache_key = f"ids_{lemma}_{lang}"
    if cache_key in runtime_api_cache: return runtime_api_cache[cache_key]

    params = {'lemma': lemma, 'searchLang': lang, 'key': BABELNET_API_KEY}
    try:
        response = requests.get(BABELNET_API_URL, params=params)
        
        if response.status_code == 200:
            data = [res['id'] for res in response.json()]
            runtime_api_cache[cache_key] = data
            return data
        else:
            return []
            
    except Exception as e:
        return []

def fetch_synset_gloss(synset_id, lang="EN"):
    cache_key = f"gloss_{synset_id}_{lang}"
    if cache_key in runtime_api_cache: return runtime_api_cache[cache_key]

    params = {'id': synset_id, 'targetLang': lang, 'key': BABELNET_API_KEY}
    try:
        response = requests.get(BABELNET_INFO_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'glosses' in data:
                for gloss in data['glosses']:
                    if gloss['language'] == lang:
                        runtime_api_cache[cache_key] = gloss['gloss']
                        return gloss['gloss']
        else:
            pass
            
    except Exception as e:
        pass
    return ""

# MODELS 

print("Loading Translation Model (NLLB)...")
tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
trans_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME).to(DEVICE)
translator = pipeline("translation", model=trans_model, tokenizer=tokenizer, src_lang="eng_Latn", tgt_lang="urd_Arab", max_length=400, device=DEVICE_ID)

print(f"Loading English WSD Model ({EN_EMBEDDING_MODEL})...")
wsd_model_en = SentenceTransformer(EN_EMBEDDING_MODEL, device=DEVICE)

print(f"Loading Urdu WSD Model ({UR_EMBEDDING_MODEL})...")
wsd_model_ur = SentenceTransformer(UR_EMBEDDING_MODEL, device=DEVICE)

print("Loading Alignment Model (SimAlign)...")
try:
    from simalign import SentenceAligner
    aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai", device=DEVICE)
    HAS_SIMALIGN = True
except ImportError:
    print("Warning: 'simalign' library not found.")
    HAS_SIMALIGN = False

# WSD LOGIC (Semantic Similarity)
def perform_wsd_embedding(target_word, context_sentence, lang_code="EN"):
    """
    Finds best Synset ID by comparing Context Sentence Embedding with Gloss Embeddings.
    """
    # Candidates
    synset_ids = fetch_babelnet_ids(target_word, lang_code)
    if not synset_ids: return None

    candidate_glosses = []
    valid_ids = []

    # Prepare Definitions (Fetch English glosses)
    for sid in synset_ids:
        gloss = fetch_synset_gloss(sid, "EN")
        if gloss:
            valid_ids.append(sid)
            candidate_glosses.append(gloss)
    
    if not valid_ids:
        return synset_ids[0] 

    # Select Model
    active_model = wsd_model_en if lang_code == "EN" else wsd_model_ur

    # Compute Embeddings
    context_embedding = active_model.encode(context_sentence, convert_to_tensor=True)
    gloss_embeddings = active_model.encode(candidate_glosses, convert_to_tensor=True)

    # Find Best Match
    cosine_scores = util.cos_sim(context_embedding, gloss_embeddings)[0]
    best_idx = torch.argmax(cosine_scores).item()
    
    return valid_ids[best_idx]

def resolve_gold_id(word, definition):
    synset_ids = fetch_babelnet_ids(word, "EN")
    if not synset_ids: return None

    candidate_glosses = []
    valid_ids = []

    for sid in synset_ids:
        gloss = fetch_synset_gloss(sid, "EN")
        if gloss:
            valid_ids.append(sid)
            candidate_glosses.append(gloss)
            
    if not valid_ids: return synset_ids[0]

    def_embedding = wsd_model_en.encode(definition, convert_to_tensor=True)
    gloss_embeddings = wsd_model_en.encode(candidate_glosses, convert_to_tensor=True)
    
    best_idx = torch.argmax(util.cos_sim(def_embedding, gloss_embeddings)[0]).item()
    return valid_ids[best_idx]

# HELPERS
def run_translation(text):
    output = translator(text)
    return output[0]['translation_text']

def run_alignment(src_text, trg_text, focus_word):
    if not HAS_SIMALIGN: return None
    try:
        alignments = aligner.get_word_aligns(src_text, trg_text)
        src_words = src_text.split()
        trg_words = trg_text.split()
        clean_focus = focus_word.strip(".,!?").lower()
        
        src_idx = -1
        for i, word in enumerate(src_words):
            if clean_focus in word.lower():
                src_idx = i
                break
        if src_idx == -1: return None

        # Prioritize MWMF (Maximum Weight Matching) from 'mai' outputs
        matches = alignments.get('mwmf', [])
        if not matches: matches = alignments.get('inter', [])
        if not matches: matches = alignments.get('itermax', [])
        
        trg_indices = [t for s, t in matches if s == src_idx]
        if not trg_indices: return None
        return " ".join([trg_words[t] for t in sorted(trg_indices)])
    except Exception:
        return None

# MAIN PROCESSING

def process_item_ids(sample_id, data):
    cached_entry = result_cache.get(sample_id)
    if cached_entry: return cached_entry

    homonym = data['homonym']
    sentence = data['sentence']
    ending = data.get('ending', '')
    judged_meaning = data['judged_meaning']
    
    example_sentence = data.get('example_sentence', '')
    full_english_sentence = f"{sentence} {ending}".strip()

    # Gold ID
    if example_sentence:
        gold_id = perform_wsd_embedding(homonym, example_sentence, "EN")
    else:
        gold_id = perform_wsd_embedding(homonym, data['judged_meaning'], "EN")

    # EN ID
    en_id = perform_wsd_embedding(homonym, full_english_sentence, "EN")

    # Example ID
    example_id = gold_id
    # if example_sentence:
    #     example_id = perform_wsd_embedding(homonym, example_sentence, "EN")

    # Urdu ID
    urdu_id = None
    urdu_translation = run_translation(full_english_sentence)
    print(urdu_translation)
    aligned_word = run_alignment(full_english_sentence, urdu_translation, homonym)
    if aligned_word:
        clean_word = aligned_word.strip(".,!؟")
        urdu_id = perform_wsd_embedding(clean_word, urdu_translation, "UR")

    result_data = {
        "id": str(sample_id),
        "homonym": homonym,
        "gold_id": gold_id,
        "en_id": en_id,
        "example_id": example_id,
        "urdu_id": urdu_id
    }
    
    if gold_id or en_id:
        result_cache.set(sample_id, result_data)
        
    return result_data

def compute_score(gold_id, en_id, urdu_id, example_id):
    score = 5.0
    if not gold_id: return 3.0
    
    if en_id != gold_id: score -= 2
    if urdu_id != gold_id: score -= 1
    if example_id is not None and example_id != gold_id: score -= 1
    if (en_id != gold_id) and (urdu_id != gold_id) and (en_id is not None) and (en_id == urdu_id):
        score -= 1
        
    return max(1.0, score)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_FILE): return

    print(f"Reading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    final_predictions = []
    print("Starting Processing...")
    
    sorted_keys = sorted(raw_data.keys(), key=lambda x: int(x))
    
    for seq_id, key in enumerate(tqdm(sorted_keys)):
        item_data = raw_data[key]
        sample_id = item_data.get('sample_id')
        if not sample_id: continue

        try:
            ids_data = process_item_ids(sample_id, item_data)
            
            score = compute_score(
                ids_data.get('gold_id'),
                ids_data.get('en_id'),
                ids_data.get('urdu_id'),
                ids_data.get('example_id')
            )
            
            prediction_entry = {"id": str(seq_id), "prediction": score}
            final_predictions.append(prediction_entry)
            
            if len(final_predictions) % 10 == 0:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    for entry in final_predictions:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
        except Exception as e:
            print(f"\nSkipping ID {seq_id} due to error: {e}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in final_predictions:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print("Done.")

if __name__ == "__main__":
    main()