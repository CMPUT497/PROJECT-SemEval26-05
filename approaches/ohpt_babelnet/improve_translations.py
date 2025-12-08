import json
import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

DATA_FILE = "../../data/dev.json"               
ANNOTATIONS_FILE = "../../Manual_Annotations/annotations.tsv" 
# OUTPUT_FOLDER = "../../predictions"
# OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "translated_sentences.jsonl")
OUTPUT_FILE = ("translated_sentences.jsonl")

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

def load_annotations(filepath):
    candidates_map = {}
    if not os.path.exists(filepath):
        print(f"Warning: Annotations file {filepath} not found.")
        return candidates_map

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        
        for row in reader:
            homonym = row.get('homonym (lemma)', '').strip().lower()
            if not homonym: continue

            if homonym not in candidates_map:
                candidates_map[homonym] = []

            w1 = row.get('F1', '').strip().replace('*', '') 
            g1 = row.get('gloss 1', '').strip()
            if w1:
                entry = f"{w1} (Meaning: {g1})"
                if entry not in candidates_map[homonym]:
                    candidates_map[homonym].append(entry)

            w2 = row.get('F2', '').strip().replace('*', '')
            g2 = row.get('gloss 2', '').strip()
            if w2:
                entry = f"{w2} (Meaning: {g2})"
                if entry not in candidates_map[homonym]:
                    candidates_map[homonym].append(entry)

    print(f"Loaded annotations for {len(candidates_map)} unique homonyms.")
    return candidates_map

def load_models():
    print("Loading NLLB (Translator)...")
    trans_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    trans_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).to(DEVICE)
    translator = pipeline("translation", model=trans_model, tokenizer=trans_tokenizer, 
                          src_lang="eng_Latn", tgt_lang="urd_Arab", max_length=512, device=0 if DEVICE=="cuda" else -1)

    print("Loading Llama-3 (Refiner)...")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    if llm_tokenizer.pad_token_id is None:
        llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
        
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )
    
    return translator, llm_tokenizer, llm_model

def refine_translation(llm_model, llm_tokenizer, source_text, draft_translation, homonym, options):
    options_text = "\n".join([f"- {opt}" for opt in options])
    
    prompt_content = f"""You are an expert Urdu translator and editor.
I have an English source text and a draft Urdu translation. The draft may use the incorrect Urdu word for the ambiguous English word "{homonym}".

**English Source:** "{source_text}"
**Draft Urdu Translation:** "{draft_translation}"

**Available Urdu Options for "{homonym}":**
{options_text}

**Task:**
1. Analyze the "English Source" to determine the specific meaning of "{homonym}" in this context.
2. Select the BEST Urdu option from the list above.
3. Rewrite the Urdu translation. Ensure the selected Urdu word is used and the sentence grammar is correct.

**Output:**
Return ONLY the final Urdu text. Do not provide explanations."""

    messages = [{"role": "user", "content": prompt_content}]
    
    input_ids = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(llm_model.device)
    
    with torch.no_grad():
        outputs = llm_model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=llm_tokenizer.pad_token_id
        )
        
    response = llm_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return


    existing_ids = set()
    if os.path.exists(OUTPUT_FILE):
        print(f"Checking existing translations in {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    existing_ids.add(record.get('id'))
                except json.JSONDecodeError:
                    continue
        print(f"Found {len(existing_ids)} items already translated.")

    candidates_map = load_annotations(ANNOTATIONS_FILE)
    translator, llm_tokenizer, llm_model = load_models()

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("\n--- Starting Translation Pipeline ---\n")
    print(f"Appending results to: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
        
        iterable = data.values() if isinstance(data, dict) else data

        for i, item in enumerate(iterable):
            
            if i in existing_ids:
                if i % 50 == 0:
                    print(f"Skipping ID {i} (Already exists)")
                continue

            homonym = item.get('homonym', "")
            
            p = item.get('precontext', '') or ""
            s = item.get('sentence', '') or ""
            e = item.get('ending', '') or ""
            full_english_text = f"{p} {s} {e}".strip()
            
            draft_out = translator(full_english_text)
            draft_urdu = draft_out[0]['translation_text']
            
            final_urdu = draft_urdu
            refined_flag = False

            if homonym.lower() in candidates_map:
                options = candidates_map[homonym.lower()]
                final_urdu = refine_translation(llm_model, llm_tokenizer, full_english_text, draft_urdu, homonym, options)
                refined_flag = True
            
            result_entry = {
                "id": i,
                "final_translation": final_urdu,
            }
            
            outfile.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            
            outfile.flush() 

            print(f"Processed ID {i} | Refined: {refined_flag}")

    print("\nProcessing complete. Results saved.")

if __name__ == "__main__":
    main()