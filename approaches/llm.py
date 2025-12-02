import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

INPUT_FILE = "../data/dev.json"
OUTPUT_FOLDER = "predictions"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "llm_predictions.jsonl")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

def setup_model():
    print(f"Loading Llama 3 model: {MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, # Llama 3 prefers bfloat16
        device_map="auto"
    )
    
    return tokenizer, model, device

def create_prompt_messages(item):
    """
    Constructs a list of messages for the Llama 3 chat template.
    """
    target_context = f"{item['precontext']} {item['sentence']} {item['ending']}"
    
    system_content = (
        "You are a semantic judge. Rate how well the homonym in the target context matches the provided definition.\n"
        "Reply ONLY with the integer score (1-5)."
    )

    # User Query (The Task)
    user_content = f"""
Definition Criteria:
- Homonym: "{item['homonym']}"
- Meaning: "{item['judged_meaning']}"
- Correct Usage Example: "{item['example_sentence']}"

Target Context to Evaluate:
"{target_context}"

On a scale of 1 to 5 (5 = perfect match, 1 = completely different), score the usage.
"""
    
    # Return structured messages
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    tokenizer, model, device = setup_model()
    
    # Define terminators for Llama 3 
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    print(f"Reading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Starting inference on {len(data)} items...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for key, item in data.items():
            # Create message structure
            messages = create_prompt_messages(item)
            
            # Apply Chat Template 
            prompt_ids = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    prompt_ids,
                    max_new_tokens=5,
                    eos_token_id=terminators,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Decode 
            response_text = tokenizer.decode(outputs[0][prompt_ids.shape[-1]:], skip_special_tokens=True)
            prediction_str = response_text.strip()
            
            # Extract Score
            digits = [s for s in prediction_str if s.isdigit()]
            if digits:
                prediction_score = int(digits[0])
            else:
                prediction_score = 1
                print(f"Warning ID {key}: Could not parse score from '{prediction_str}'. Defaulting to 1.")

            # Save
            result_obj = {"id": key, "prediction": prediction_score}
            outfile.write(json.dumps(result_obj) + "\n")
            
            if int(key) % 10 == 0:
                print(f"Processed ID {key}: Predicted {prediction_score}")

    print("Processing complete.")

if __name__ == "__main__":
    main()