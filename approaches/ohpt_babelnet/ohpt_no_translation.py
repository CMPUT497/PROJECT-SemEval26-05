import json
import os
import requests
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION ---
INPUT_FILE = "../../data/dev.json"
TRANSLATION_FILE = "../../predictions/translated_sentences.jsonl"
EXISTING_CACHE_FILE = "processing_cache.json"      # Read from here (EN/Gold IDs)
NEW_CACHE_FILE = "updated_processing_cache.json"   # Write to here (Adds Urdu IDs)
OUTPUT_DIR = "../../predictions"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ohpt_manual_tranlations.jsonl")

# API CONFIG
BABELNET_API_KEY = "9dce12de-cd8d-406c-a219-e8e2a77275cc" 
BABELNET_API_URL = "https://babelnet.io/v9/getSynsetIds"
BABELNET_INFO_URL = "https://babelnet.io/v9/getSynset"

# WSD MODELS
UR_EMBEDDING_MODEL = "xlm-roberta-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# --- 1. API & MODEL HELPERS ---
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
    except Exception:
        pass
    return []

def fetch_synset_gloss(synset_id, lang="EN"):
    # We always fetch English glosses for semantic comparison
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
    except Exception:
        pass
    return ""

print(f"Loading Urdu WSD Model...")
wsd_model_ur = SentenceTransformer(UR_EMBEDDING_MODEL, device=DEVICE)

print("Loading Alignment Model (SimAlign)...")
try:
    from simalign import SentenceAligner
    aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai", device=DEVICE)
    HAS_SIMALIGN = True
except ImportError:
    print("Warning: 'simalign' library not found. Alignment will fail.")
    HAS_SIMALIGN = False

# --- 2. CORE LOGIC ---

def run_alignment(src_text, trg_text, focus_word):
    if not HAS_SIMALIGN or not trg_text: return None
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

        matches = alignments.get('mwmf', []) or alignments.get('inter', []) or alignments.get('itermax', [])
        trg_indices = [t for s, t in matches if s == src_idx]
        
        if not trg_indices: return None
        return " ".join([trg_words[t] for t in sorted(trg_indices)])
    except Exception:
        return None

def perform_urdu_wsd(urdu_word, urdu_context):
    """
    1. Fetches BabelNet IDs for the Urdu word (API CALL).
    2. Fetches English glosses for those IDs (API CALL).
    3. Compares Urdu Context embedding to English Gloss embeddings.
    """
    synset_ids = fetch_babelnet_ids(urdu_word, "UR")
    if not synset_ids: return None

    candidate_glosses = []
    valid_ids = []

    for sid in synset_ids:
        gloss = fetch_synset_gloss(sid, "EN")
        if gloss:
            valid_ids.append(sid)
            candidate_glosses.append(gloss)
    
    if not valid_ids: return synset_ids[0]

    context_embedding = wsd_model_ur.encode(urdu_context, convert_to_tensor=True)
    gloss_embeddings = wsd_model_ur.encode(candidate_glosses, convert_to_tensor=True)

    cosine_scores = util.cos_sim(context_embedding, gloss_embeddings)[0]
    best_idx = torch.argmax(cosine_scores).item()
    
    return valid_ids[best_idx]

def compute_score(gold_id, en_id, urdu_id):
    score = 5.0
    if not gold_id: return 3.0
    
    if en_id and en_id != gold_id: score -= 2
    if urdu_id and urdu_id != gold_id: score -= 1
    if (en_id != gold_id) and (urdu_id != gold_id) and (en_id == urdu_id):
        score -= 1
        
    return max(1.0, score)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- LOAD RESOURCES ---
    print(f"Reading Data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"Reading Translations from {TRANSLATION_FILE}...")
    translations_map = {}
    if os.path.exists(TRANSLATION_FILE):
        with open(TRANSLATION_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    translations_map[rec['id']] = rec['final_translation']
                except: continue

    print(f"Reading Existing Cache from {EXISTING_CACHE_FILE}...")
    id_cache = {}
    if os.path.exists(EXISTING_CACHE_FILE):
        with open(EXISTING_CACHE_FILE, 'r', encoding='utf-8') as f:
            id_cache = json.load(f)
    else:
        print("Existing cache not found. Script requires cached EN/Gold IDs.")
        return

    # Dictionary to store the NEW complete cache (Old IDs + New Urdu IDs)
    new_cache_data = {}
    
    final_predictions = []
    print("\n--- Starting Processing & Caching ---")

    sorted_keys = sorted(raw_data.keys(), key=lambda x: int(x))

    for seq_id, key in enumerate(tqdm(sorted_keys)):
        item = raw_data[key]
        sample_id = str(item.get('sample_id'))
        homonym = item.get('homonym')
        
        # 1. RETRIEVE EXISTING IDs (No API Calls)
        cached_entry = id_cache.get(sample_id)
        if not cached_entry:
            # Skip if we don't have the base data
            final_predictions.append({"id": str(seq_id), "prediction": 3.0})
            continue

        gold_id = cached_entry.get('gold_id')
        en_id = cached_entry.get('en_id')
        example_id = cached_entry.get('example_id')

        # 2. COMPUTE URDU ID (API Calls happen here)
        urdu_id = None
        translation_text = translations_map.get(seq_id)
        
        if translation_text:
            full_english_text = f"{item.get('sentence', '')} {item.get('ending', '')}"
            
            # Align
            aligned_urdu_word = run_alignment(full_english_text, translation_text, homonym)
            
            # WSD
            if aligned_urdu_word:
                clean_urdu = aligned_urdu_word.strip(".,!؟")
                urdu_id = perform_urdu_wsd(clean_urdu, translation_text)
        
        # 3. COMPUTE SCORE
        score = compute_score(gold_id, en_id, urdu_id)
        final_predictions.append({"id": str(seq_id), "prediction": score})
        
        # 4. UPDATE CACHE DATA
        new_cache_data[sample_id] = {
            "id": sample_id,
            "homonym": homonym,
            "gold_id": gold_id,
            "en_id": en_id,
            "example_id": example_id,
            "urdu_id": urdu_id  # This is the new value we just computed
        }
        
        # Periodic Save of Predictions
        if len(final_predictions) % 20 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                for entry in final_predictions:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            # Periodic Save of Cache (to save progress if script crashes)
            with open(NEW_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(new_cache_data, f, ensure_ascii=False, indent=4)

    # --- FINAL SAVES ---
    
    # Save Predictions
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in final_predictions:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Save New Cache
    print(f"Saving updated cache with Urdu IDs to {NEW_CACHE_FILE}...")
    with open(NEW_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_cache_data, f, ensure_ascii=False, indent=4)
            
    print("Processing Complete.")

if __name__ == "__main__":
    main()