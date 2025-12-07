import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

INPUT_FILE = "../data/dev.json"
OUTPUT_FOLDER = "../predictions"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "llm_judge_predictions.jsonl")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

def setup_model():
    # CHECK FOR CUDA
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU.")

    print(f"Loading Llama 3 model: {MODEL_ID}...")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" 
    )
    
    return tokenizer, model, device

def create_prompt_messages(item):
    """
    Constructs the prompt exactly as described in Appendix B.3 of the paper.
    """
    story_text = f"{item['precontext']} **{item['sentence']}** {item.get('ending', '')}"
    
    user_content = f"""You will see a short text in which one sentence is marked with "**". That sentence contains a word that can typically take on multiple different meanings, depending on the context. One of those meanings is given to you.

**Your task is simple: Annotate how plausible a meaning of a word is in the context of the short text using one of five scores:**

***1**: The displayed meaning is not plausible at all given the context.
***2**: The displayed meaning is theoretically conceivable, but less plausible than other meanings.
***3**: The displayed meaning represents one of multiple, similarly plausible interpretations.
***4**: The displayed meaning represents the most plausible interpretation; other meanings may still be conceivable.
***5**: The displayed meaning is the only plausible meaning given the context.

There will be times where there is no objectively correct answer. Whatever the case, always look at all of the sentences and carefully think about how plausible each meaning would be.

Now take a look at the following text: {story_text}

In this context, how plausible is it that the meaning of the word "{item['homonym']}" is "{item['judged_meaning']}"?

Return only the numbered score (1, 2, 3, 4 or 5). Do not return anything else!"""
    
    return [
        {"role": "user", "content": user_content}
    ]

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    tokenizer, model, device = setup_model()
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    print(f"Reading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Starting inference on {len(data)} items...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        iterable = data.items() if isinstance(data, dict) else enumerate(data)

        for key, item in iterable:
            messages = create_prompt_messages(item)
            
            # Apply Chat Template 
            prompt_ids = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(device) # Ensure inputs are moved to CUDA
            
            attention_mask = (prompt_ids != tokenizer.pad_token_id).long()

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=5,
                    eos_token_id=terminators,
                    do_sample=False
                )
            
            response_text = tokenizer.decode(outputs[0][prompt_ids.shape[-1]:], skip_special_tokens=True)
            prediction_str = response_text.strip()
            
            digits = [s for s in prediction_str if s.isdigit()]
            if digits:
                prediction_score = int(digits[0])
            else:
                prediction_score = 1
                print(f"Warning ID {key}: Could not parse score from '{prediction_str}'. Defaulting to 1.")

            result_obj = {"id": key, "prediction": prediction_score}
            outfile.write(json.dumps(result_obj) + "\n")
            
            if isinstance(key, int) and key % 10 == 0:
                print(f"Processed item {key}...")
            elif isinstance(key, str) and int(key) % 10 == 0:
                print(f"Processed item {key}...")

    print("Processing complete.")

if __name__ == "__main__":
    main()