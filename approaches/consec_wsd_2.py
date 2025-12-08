import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from nltk.stem import PorterStemmer
import torch.nn.functional as F
import pytorch_lightning.callbacks



class WSDSimpleDataset(Dataset):
    """
    Each item returns:
    - context: string with <d> ... </d> marking target word
    - glosses: list of gloss strings
    - labels: either:
        (A) one-hot vector → hard label
        (B) probability distribution → soft label
    - label_type: "hard" or "soft"
    """
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ConSeCModel(nn.Module):
    """
    Transformer over:
        [context with <d>] + [<def> gloss1] + [<def> gloss2] + ...

    Output → logits for each gloss candidate.
    """
    def __init__(self, model_name="/home/girigowd/consec/experiments/released-ckpts"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add ConSeC markers
        special_tokens = ["<d>", "</d>", "<def>"]
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        hidden = self.encoder.config.hidden_size
        
        # Simple linear classifier head → one logit per gloss
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, context, glosses):
        """
        context: string with markers
        glosses: list of gloss strings
        """
        combined = context
        def_positions = []
        
        context_tokens = self.tokenizer(context, add_special_tokens=False)['input_ids']
        current_pos = len(context_tokens) + 1  # +1 for [CLS]
        
        for g in glosses:
            gloss_text = f" <def> {g}"
            combined += gloss_text
            
            def_token_id = self.tokenizer.convert_tokens_to_ids('<def>')
            def_positions.append(current_pos)
            
            gloss_tokens = self.tokenizer(gloss_text, add_special_tokens=False)['input_ids']
            current_pos += len(gloss_tokens)
        
        enc = self.tokenizer(
            combined,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.encoder.device)
        
        outputs = self.encoder(**enc)
        hidden_states = outputs.last_hidden_state
        
        # Extract representation at each <def> token position
        gloss_representations = []
        for pos in def_positions:
            if pos < hidden_states.size(1):
                gloss_representations.append(hidden_states[0, pos, :])
        
        gloss_reps = torch.stack(gloss_representations)
        
        logits = self.classifier(gloss_reps)
        
        return logits
  

def wsd_collate_fn(batch):
    contexts = [item["context"] for item in batch]
    glosses = [item["glosses"] for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.float) for item in batch]
    label_types = [item["label_type"] for item in batch]

    return {
        "contexts": contexts,
        "glosses": glosses,
        "labels": labels,
        "label_types": label_types,
    }

def soft_label_loss(pred_logits, soft_targets):
    log_probs = F.log_softmax(pred_logits, dim=-1)
    return F.kl_div(log_probs, soft_targets, reduction="batchmean")

def hard_label_loss(pred_logits, hard_targets):
    logits = pred_logits.squeeze(-1)
    return F.cross_entropy(logits, hard_targets)

def train_consec(model, dataloader, optimizer, device, epochs=3, save_path="best_consec_model_2.pt"):
    model.train()
    model.to(device)

    best_loss = float('inf')
    best_epoch = -1

    for epoch in range(epochs):
        total_loss = 0

        for batch in dataloader:
            contexts = batch["contexts"]
            glosses_batch = batch["glosses"]
            labels_batch = batch["labels"]
            kinds = batch["label_types"]

            # Process examples one-by-one
            for context, glosses, labels, kind in zip(contexts, glosses_batch, labels_batch, kinds):
                labels = labels.to(device)
                # print()
                # print(context)
                # print("------------------")
                # print(labels)
                # print("------------------")
                # print(glosses)
                # print("------------------")
                # print(kind)
                # print("------------------")
                
                optimizer.zero_grad()

                logits = model(context, glosses)

                if kind == "soft":
                    loss = soft_label_loss(logits, labels)
                else:  # "hard"
                    loss = hard_label_loss(logits, labels)
                # print("LOSS: :", loss)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss:.4f} ")

        if total_loss < best_loss:
            best_loss = total_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"  [Best model saved at epoch {best_epoch} with loss {best_loss:.4f}]")

def consec_infer(model, context, glosses, device="cpu", top_k=None, save_path="best_consec_model_2.pt"):
    """
    model: trained ConSeCModel
    context: string (with <d> markers)
    glosses: list of gloss strings
    top_k: optional integer for top-k senses
    """
    model.load_state_dict(torch.load(save_path))
    
    model.eval()
    model.to(device)

    with torch.no_grad():
        logits = model(context, glosses) 
        probs = F.softmax(logits, dim=-1)

    # Convert to CPU for printing or further use
    logits = logits.cpu().numpy()
    probs = probs.cpu().numpy()

    # Get best gloss index
    best_idx = probs.argmax()
    best_gloss = glosses[best_idx]

    # Optional: top-k glosses
    if top_k is not None:
        top_indices = probs.argsort()[::-1][:top_k]
        top_results = [(glosses[i], float(probs[i])) for i in top_indices]
    else:
        top_results = None
    
    # print("***************************")
    # print("context: ", context)
    # print("***************************")
    # print("glosses: ", glosses)
    # print("***************************")
    # print("probabilities: ", probs)
    # print()
    
    return {
        "context": context,
        "glosses": glosses,
        "logits": logits,
        "probs": probs,
        "best_index": int(best_idx),
        "best_gloss": best_gloss,
        "top_k": top_results
    }

def run_inference(model, path, output_file, device):
    
    with open(path, 'r', encoding='utf8') as f:
        file_dict = json.load(f)

    all_homonym_instances = {}    
    for id, item in file_dict.items():
        homonym = item['homonym']
        if homonym not in all_homonym_instances:
            all_homonym_instances[homonym] = {}
            all_homonym_instances[homonym]["contexts"] = []
            all_homonym_instances[homonym]["glosses"] = []
            all_homonym_instances[homonym]["instances"] = []

        sentence = item['sentence']
        hom_idx = sentence.index(homonym)
        sentence = sentence[:hom_idx] + "<d> " + homonym + " </d>" + sentence[hom_idx+len(homonym):]
        ending = item['ending']
        if ending != "":
            context = item['precontext'] + " " + sentence + " " + item['ending']
        else:
            context = item['precontext'] + " " + sentence
        
        if context not in all_homonym_instances[homonym]["contexts"]:
            all_homonym_instances[homonym]["contexts"].append(context)
        context_idx = all_homonym_instances[homonym]["contexts"].index(context)
        
        gloss = item['judged_meaning']
        if gloss not in all_homonym_instances[homonym]["glosses"]:
            all_homonym_instances[homonym]["glosses"].append(gloss)
        gloss_idx = all_homonym_instances[homonym]["glosses"].index(gloss)
        
        all_homonym_instances[homonym]["instances"].append({
            "id": id,
            "context_idx": context_idx,
            "gloss_idx": gloss_idx
        })

    inference_results = {}
    for homonym, data in all_homonym_instances.items():
        inference_results[homonym] = {
            "contexts": data["contexts"],
            "glosses": data["glosses"],
            "context_results": [],
            "instance_results": []
        }
        # Run inference for each unique context
        context_predictions = []
        for context_idx, context in enumerate(data["contexts"]):
            glosses = data["glosses"]
            res = consec_infer(model, context, glosses, device=device, top_k=3, save_path="best_consec_model_2.pt")
            context_predictions.append(res)
            inference_results[homonym]["context_results"].append({
                "context_idx": context_idx,
                "predictions": res
            })
        # Map predictions to each instance/ID
        for instance in data["instances"]:
            id = instance["id"]
            context_idx = instance["context_idx"]
            gloss_idx = instance["gloss_idx"]
            # Get the prediction for this context
            prediction = context_predictions[context_idx]
            inference_results[homonym]["instance_results"].append({
                "id": id,
                "context_idx": context_idx,
                "gloss_idx": gloss_idx,
                "true_gloss": data["glosses"][gloss_idx],
                "predictions": prediction["probs"][gloss_idx]
            })

    with open(output_file, 'w', encoding='utf8') as outf:
        flat_results = []
        for homonym, data in inference_results.items():
            for instance in data["instance_results"]:
                prediction = instance['predictions']
                flat_results.append({
                    "id": str(instance["id"]),
                    "prediction": int(prediction*5)
                })
        for item in flat_results:
            outf.write(json.dumps(item) + "\n")
    
    print("OUTPUT SAVED!! TASK COMPLETE")        

def prepare_training_dataset(path):

    with open(path, 'r', encoding='utf8') as f:
        file_dict = json.load(f)

    all_homonym_instances = {}
    for id, item in file_dict.items():
        homonym = item['homonym']
        if homonym not in all_homonym_instances:
            all_homonym_instances[homonym] = {
                "contexts": [],
                "glosses": [],
                "soft_labels": {},
                "soft_labels_val": {},
                "hard_labels": {},
                "label_type": {}
            }

        gloss = item['judged_meaning']
        if gloss not in all_homonym_instances[homonym]["glosses"]:
            all_homonym_instances[homonym]["glosses"].append(gloss)

        stemmer = PorterStemmer()
        sentence = item['sentence']
        if homonym in sentence:
            idx = sentence.index(homonym)
        else:
            stem = stemmer.stem(homonym)
            if stem in sentence:
                idx = sentence.index(stem)
            else:
                continue

        sentence = sentence[:idx] + f"<d> {homonym} </d>" + sentence[idx+len(homonym):]
        ending = item["ending"]

        if ending:
            soft_context = item["precontext"] + " " + sentence + " " + ending
        else:
            soft_context = item["precontext"] + " " + sentence

        if soft_context not in all_homonym_instances[homonym]["contexts"]:
            all_homonym_instances[homonym]["contexts"].append(soft_context)
            all_homonym_instances[homonym]["soft_labels"][soft_context] = []
            all_homonym_instances[homonym]["soft_labels_val"][soft_context] = {}
            all_homonym_instances[homonym]["label_type"][soft_context] = "soft"

        all_homonym_instances[homonym]["soft_labels"][soft_context].append(gloss)
        all_homonym_instances[homonym]["soft_labels_val"][soft_context][gloss] = item['average']
        
        ex_sent = item["example_sentence"]
        if homonym in ex_sent:
            idx = ex_sent.index(homonym)
        else:
            stem = stemmer.stem(homonym)
            if stem in ex_sent:
                idx = ex_sent.index(stem)
            else:
                continue
        ex_sent = ex_sent[:idx] + f"<d> {homonym} </d>" + ex_sent[idx+len(homonym):]
        hard_context = ex_sent
        if hard_context not in all_homonym_instances[homonym]["contexts"]:
            all_homonym_instances[homonym]["contexts"].append(hard_context)
            all_homonym_instances[homonym]["hard_labels"][hard_context] = gloss
            all_homonym_instances[homonym]["label_type"][hard_context] = "hard"

    dataset = []
    for homonym, data in all_homonym_instances.items():
        glosses = data["glosses"]

        for ctx in data["contexts"]:
            ltype = data["label_type"][ctx]

            if ltype == "soft":
                gloss_hist = [0] * len(glosses)
                for p, g in enumerate(data["soft_labels"][ctx]):
                    gloss_idx = glosses.index(g)
                    gloss_hist[gloss_idx] += data["soft_labels_val"][ctx][g]
                total = sum(gloss_hist)
                if total == 0:
                    label_vec = [1/len(glosses)] * len(glosses)
                else:
                    label_vec = [x / total for x in gloss_hist]
                # print("################")
                # print(ctx)
                # print("################")
                # print(glosses)
                # print("################")
                # print(label_vec)
                # print("################")
                # print()
                dataset.append({
                    "context": ctx,
                    "glosses": glosses.copy(),
                    "labels": label_vec,
                    "label_type": "soft"
                })
            else:
                g = data["hard_labels"][ctx]
                gloss_idx = glosses.index(g)
                onehot = [0] * len(glosses)
                onehot[gloss_idx] = 1

                dataset.append({
                    "context": ctx,
                    "glosses": glosses.copy(),
                    "labels": onehot,
                    "label_type": "hard"
                })

    return dataset

def main():
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    ckpt_path = "../consec/experiments/released-ckpts/consec_wngt_best.ckpt"

    # 1. Load checkpoint (trusted source → weights_only=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 2. Create the model EXACTLY like the original ConSeC code does
    model = ConSeCModel(model_name="microsoft/deberta-v3-large")

    # 3. Remove the prefix "sense_extractor.model." that Lightning adds
    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("sense_extractor.model."):
            new_key = k.replace("sense_extractor.model.", "")
        elif k.startswith("sense_extractor."):
            new_key = k.replace("sense_extractor.", "")
        else:
            new_key = k
        state_dict[new_key] = v

    # 4. Load weights + move to GPU
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Model loaded perfectly – running inference")
    run_inference(model, "data/dev.json", "predictions/consec_wsd_2_predictions.JSONL", device)


if __name__ == "__main__":
    main()
