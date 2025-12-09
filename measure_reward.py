from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaTokenizer, LlamaForSequenceClassification, AutoModel
import argparse
import torch
import json
import re
import time
import numpy as np
from tqdm import tqdm
np.random.seed(42)
torch.manual_seed(42)


def main(args):
    with open(args.out_file, "r") as out_f:
        database = json.load(out_f)

    start = time.time()
    model_name = args.model_name
    print(f"Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    rm_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1,
                                                                  torch_dtype=torch.float16).to(args.device)
    rm_model = rm_model.to(args.device)
    rm_model.eval()
    print(f"model loaded in : {time.time() - start :.2f}")


    rm_scores = []
    num_skip = 0
    for line in tqdm(database):
        if line == "run_configs":
            continue
        if type(database[line]['response']) == dict:
            last_response_key = list(database[line]['response'])[-1]
            last_response = database[line]['response'][last_response_key]
            last_response = last_response['worker']
        else:
            last_response = database[line]['response']
        combined_out = database[line]['prompt'] + last_response
        # get rm_score
        tokens = tokenizer(combined_out, return_tensors="pt", padding=True).input_ids.to(args.device)
        # print(f"{tokens.shape=}")
        if tokens.shape[1] >= 1334: return None
        rm_out = rm_model(tokens)
        rm_score = rm_out.logits.flatten().item()
        del rm_out
        del tokens

        if rm_score is None:
            print("skipped one")
            num_skip += 1
            continue
        else: rm_scores.append(rm_score)

        if len(rm_scores) >= args.num_entries:
            break

    print(f"{np.mean(rm_scores)=}")
    print(f"{num_skip=}")
    return np.mean(rm_scores)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="usvsnsp/pythia-6.9b-rm-full-hh-rlhf")
    parser.add_argument("--out_file", type=str, default="run_outs/llama_llm_only_direct_0.jsonl")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_entries", type=int, default=50)

    args = parser.parse_args()

    main(args)