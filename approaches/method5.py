import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def contextual_embedding_similarity(path):

    with open(path, 'r', encoding='utf8') as f:
        file_dict = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("Kingsoft-LLM/QZhou-Embedding", torch_dtype=torch.float16, padding_side="left")
    model = AutoModel.from_pretrained("Kingsoft-LLM/QZhou-Embedding", torch_dtype=torch.float16)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.gradient_checkpointing_enable()

    results = {}

    for k, v in file_dict.items():
        precontext = v.get("precontext", "")
        sentence = v.get("sentence", "")
        ending = v.get("ending", "")
        example_sentence = v.get("example_sentence", "")
        avg_score = v.get("average", "")
        homonym = v.get("homonym", "")

        big_sentence_parts = [precontext.strip(), sentence.strip(), ending.strip()]
        big_sentence = " ".join([part for part in big_sentence_parts if part])
        
        inputs_big = tokenizer(big_sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs_example = tokenizer(example_sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)

        inputs_big = {k: v.to(device) for k, v in inputs_big.items()}
        inputs_example = {k: v.to(device) for k, v in inputs_example.items()}

        with torch.no_grad():
            emb_big = model(**inputs_big)
            emb_example = model(**inputs_example)

        emb_big_vec = emb_big.last_hidden_state[0]
        tokens = tokenizer.convert_ids_to_tokens(inputs_big["input_ids"][0])
        indices = []
        for i, tok in enumerate(tokens):
            tok = tok.lstrip("Ġ")
            if homonym == tok.lower():
                indices.append(i)
        if not indices:
            from nltk.stem import PorterStemmer
            ps = PorterStemmer()
            homonym_stem = ps.stem(v["homonym"].lower())
            for i, tok in enumerate(tokens):
                tok = tok.lstrip("Ġ")
                if ps.stem(tok.lower()) == homonym_stem:
                    indices.append(i)
                elif tok.lower() == homonym_stem:
                    indices.append(i)
        if not indices:
            homonym_stem = v["homonym"].lower()[:4]
            for i, tok in enumerate(tokens):
                tok = tok.lstrip("Ġ")
                if tok.lower().startswith(homonym_stem):
                    indices.append(i)
        selected = emb_big_vec[indices]
        emb_big_vec = selected.mean(dim=0)
        emb_big_vec = F.normalize(emb_big_vec, p=2, dim=0)
        
        print("TOKENS FOR SENTENCE: ", tokens)
        print("----------------------")
        print("INDICES: ", indices)
        print("*****************")

        emb_example_vec = emb_example.last_hidden_state[0]
        tokens = tokenizer.convert_ids_to_tokens(inputs_example["input_ids"][0])
        indices = []
        for i, tok in enumerate(tokens):
            tok = tok.lstrip("Ġ")
            if homonym == tok.lower():
                indices.append(i)
        if not indices:
            from nltk.stem import PorterStemmer
            ps = PorterStemmer()
            homonym_stem = ps.stem(v["homonym"].lower())
            for i, tok in enumerate(tokens):
                tok = tok.lstrip("Ġ")
                if ps.stem(tok.lower()) == homonym_stem:
                    indices.append(i)
                elif tok.lower() == homonym_stem:
                    indices.append(i)
        if not indices:
            homonym_stem = v["homonym"].lower()[:4]
            for i, tok in enumerate(tokens):
                tok = tok.lstrip("Ġ")
                if tok.lower().startswith(homonym_stem):
                    indices.append(i)
        selected = emb_example_vec[indices]
        emb_example_vec = selected.mean(dim=0)
        emb_example_vec = F.normalize(emb_example_vec, p=2, dim=0)

        cosine_sim = torch.dot(emb_example_vec, emb_example_vec).item()
        
        print("TOKENS FOR EXMAPLE SENTENCE: ", tokens)
        print("----------------------")
        print("INDICES: ", indices)
        print("*****************")
        

        results[k] = {
            "cosine_similarity": cosine_sim,
            "example_sentence": example_sentence,
            "big_sentence": big_sentence,
            "avg": avg_score
        }

    # You can return or print results, or save if desired
    # For now, just print the first 5 for inspection
    for k in list(results.keys()):
        print(results[k]['example_sentence'])
        print("*****************")
        print(results[k]['big_sentence'])
        print("*****************")
        print(f"Sample {k}: cosine similarity = {results[k]['cosine_similarity']:.3f}, avg_score = {float(results[k]['avg']):.3f}")
        print()
    

def main():
    device = "cuda:3" if torch.cuda.is_available() else "cpu"

    train_path = "data/train.json"
    dev_path = "data/dev.json"
    trail_path = "data/trail.json"
    output_path = "predictions/consec_wsd_predictions.JSONL"

    contextual_embedding_similarity(trail_path)
    


if __name__ == "__main__":
    main()
