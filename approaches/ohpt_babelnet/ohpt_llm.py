import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

INPUT_FILE = "../../data/dev.json"
# CACHE_FILE = "processing_cache.json" 
CACHE_FILE = "updated_processing_cache.json"
OUTPUT_FOLDER = "../../predictions"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "ohpt_llm_judge.jsonl")



MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

def setup_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    print(f"Loading Model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    
    return tokenizer, model, device

def create_augmented_prompt(item, cache_entry):
    """
    Creates a prompt that combines the story text with 'Expert Hints' 
    derived from the cached WSD IDs.
    """
    story_text = f"{item['precontext']} **{item['sentence']}** {item.get('ending', '')}"
    
    en = "Uncertain"
    ur = "Uncertain"
    gold = "Uncertain"

    if cache_entry:
        gold = cache_entry.get('gold_id')
        en = cache_entry.get('en_id')
        ur = cache_entry.get('urdu_id')
        
   
    user_content = f"""You will see a short text in which one sentence is marked with "**". That sentence contains a word that can typically take on multiple different meanings, depending on the context using provided automated analysis as a hint. 

**Your task is simple: Annotate how plausible a meaning of a word is in the context of the short text using one of five scores:**

***1**: The displayed meaning is not plausible at all given the context.
***2**: The displayed meaning is theoretically conceivable, but less plausible than other meanings.
***3**: The displayed meaning represents one of multiple, similarly plausible interpretations.
***4**: The displayed meaning represents the most plausible interpretation; other meanings may still be conceivable.
***5**: The displayed meaning is the only plausible meaning given the context.

There will be times where there is no objectively correct answer. Whatever the case, always look at all of the sentences and carefully think about how plausible each meaning would be.
**Automated Analysis Reports:**
- Gold Babelnet ID: {gold}
- English Babelnet ID: {en}.
- Urdu Babelnet ID: {ur}.

Now take a look at the following text: {story_text}

In this context, how plausible is it that the meaning of the word "{item['homonym']}" is "{item['judged_meaning']}"?

Return only the numbered score (1, 2, 3, 4 or 5). Do not return anything else!"""
    


    return [{"role": "user", "content": user_content}]

def main():
    if not os.path.exists(INPUT_FILE) or not os.path.exists(CACHE_FILE):
        print("Error: Input or Cache file not found.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    tokenizer, model, device = setup_model()
    
    print(f"Loading Data...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
        
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
        
    print(f"Processing {len(dev_data)} items using cached signals...")
    
    # Handle list vs dict format for dev_data
    items = dev_data.items() if isinstance(dev_data, dict) else enumerate(dev_data)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for index, (key, item) in enumerate(items):
            
            sample_id = str(key)
            if 'sample_id' in item:
                sample_id = str(item['sample_id'])
            
            cache_entry = cache_data.get(sample_id)
            messages = create_augmented_prompt(item, cache_entry)
            
            prompt_ids = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(device)
            
            attention_mask = (prompt_ids != tokenizer.pad_token_id).long()
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=5,
                    do_sample=False
                )
            
            resp_text = tokenizer.decode(outputs[0][prompt_ids.shape[-1]:], skip_special_tokens=True).strip()
            
            digits = [s for s in resp_text if s.isdigit()]
            score = int(digits[0]) if digits else 1
            
            res = {"id": str(index), "prediction": score}
            outfile.write(json.dumps(res) + "\n")
            
            if index % 20 == 0:
                print(f"Processed item {index} (Sample ID: {sample_id}): Predicted {score}")

    print("Done.")

if __name__ == "__main__":
    main()