from datasets import load_dataset
import argparse
import json
import yaml
from pathlib import Path
from tqdm import tqdm

from direct import TQ_direct
from indirect import TQ_indirect
from collaborative_indirect import AgenticTQ
import time
import pickle
import torch
import numpy as np
import os
np.random.seed(42)
torch.manual_seed(42)

def check_valid_args(args):
    if args.recover:
        print("LOOKS LIKE YOU WANT TO RECOVER SOME RESULTS,")
        print("MAKE SURE ALL COMMANDLINE ARGS ARE EXACTLY THE SAME!!!")
        input("PRESS ENTER TO CONTINUE")

    if not (args.max_new_token > 0):
        print("ERROR: Max tokens should be greater than 0!")
        exit(1)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print("ERROR: Config doesn't exist!")
        exit(1)
    args.out_file = f"{args.out_file}_{args.task_type}"
    out_path = Path(args.out_file + f"_0.jsonl")
    if out_path.exists() and (not args.recover):
        print("ERROR: out_path already exists!")
        exit(1)

    if not out_path.exists() and args.recover:
        print("ERROR: out_path DOESN'T exist!")
        exit(1)

    # with open(cfg_path) as f:
    #     run_configs = [json.loads(line) for line in f.readlines()]

    with open(cfg_path, 'r') as f:
        run_configs = yaml.load(f, Loader=yaml.SafeLoader)
    run_configs['dataset'] = args.dataset
    run_configs['split'] = args.split
    run_configs['run_percent'] = args.run_percent
    run_configs['max_new_token'] = args.max_new_token
    run_configs['llm_gpu'] = args.llm_gpu
    run_configs['rm_gpu'] = args.rm_gpu
    run_configs['rm2_gpu'] = args.rm2_gpu
    run_configs['config'] = args.config
    run_configs['task_type'] = args.task_type

    # validate configs
    if "rm_weight" not in run_configs:
        print(f"Missing key 'rm_weight' in {run_configs}")
        exit(1)
    elif "topk" not in run_configs:
        print(f"Missing key 'topk' in {run_configs}")
        exit(1)
    elif "mode" not in run_configs:
        print(f"Missing key 'mode' in {run_configs}")
        exit(1)
    elif "sample_temp" not in run_configs:
        print(f"Missing key 'sample_temp' in {run_configs}")
        exit(1)

    print(f"[ !! ] Please Double Check: task type {args.task_type}, loading config from {args.config}")
    return run_configs

def runprompt(search, prompt: str, rm_weight=0., topk=5, new_token=24, mode="p_sigmoid_mixing", sample_temp=None, llm_dev:str="cuda:0",
              debug=True, align=True):
    tokens, scores = search.generate(prompt, method=mode, topk=topk, max_new_token=new_token, weight=rm_weight, align=align, debug=debug)

    # too long seqlen
    if tokens is None: return None, None

    if type(tokens) == dict:
        # only collaborative TQ gives out dict that is already in text form...
        return tokens, 0, scores
    tokens_text = search.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    text_response = tokens_text.removeprefix(prompt)
    return text_response, tokens, scores

def main(args):
    run_configs = check_valid_args(args)
    config_num=0
    json_save_path = f"{args.out_file}_{config_num}.jsonl"
    hf_cache=None
    # print(f"Loaded {len(run_configs)} run configs.")
    print(f"{run_configs=}")

    print(f"Loading dataset ({args.dataset=}, {args.split=})")
    test_ds = load_dataset(args.dataset, split=args.split)
    if args.dataset == "Dahoas/full-hh-rlhf":
        # FOR HHRLHF
        test_ds = test_ds["prompt"]

    elif args.dataset == "openbmb/UltraFeedback":
        print("Running openbmb/UltraFeedback")
        test_ds = test_ds["instruction"]

    elif args.dataset == "berkeley-nest/Nectar":
        print("Running berkeley-nest/Nectar")
        test_ds = test_ds["prompt"]


    end_idx = int(len(test_ds) * (args.run_percent/100.))
    print(f"{end_idx=}, {len(test_ds)=}")

    truncated_ds = test_ds[0:end_idx]
    if args.task_type == "direct":
        print(f"Loading models ({run_configs['llm']}, {run_configs['rm']})")
        search = TQ_direct(llm_path=run_configs['llm'], rm_path=run_configs['rm'],
                           llm_device=args.llm_gpu, rm_device=args.rm_gpu)
    elif args.task_type == "indirect":
        print(f"Loading models ({run_configs['llm']}, {run_configs['rm']})")
        search = TQ_indirect(llm_path=run_configs['llm'], rm_path=run_configs['rm'],
                             llm_device=args.llm_gpu, rm_device=args.rm_gpu)
    elif args.task_type == "collab":
        print(f"Loading models ({run_configs['llm']}, {run_configs['rm']})")
        search = AgenticTQ(llm_path=run_configs['llm'], rm_path=run_configs['rm'],
                           rm2_path=run_configs['rm2'], llm_device=args.llm_gpu,
                           rm_dev=args.rm_gpu, rm2_dev=args.rm2_gpu,
                           max_iters=run_configs['max_iters'])
    else:
        print(f"ERROR, unknown task type {args.task_type}")
        exit()

    print(f"{len(truncated_ds)=}")

    data = {"run_configs": run_configs}
    if args.recover and Path(args.out_file + f"_{config_num}.jsonl").exists():
        print(f"Run already exists, checking if it's done")
        resfile = open(Path(args.out_file + f"_{config_num}.jsonl"))
        samples = resfile.readlines()

        if samples[-1] != "":
            print("last line not empty??")
            exit(1)

        last_obj = json.loads(samples[-2])
        if last_obj["prompt"] != truncated_ds[len(samples) -1]:
            print(f"PROMPTS DID NOT MATCH RECOVERY FAILED!!!")
            exit(1)

    score_overall = []
    if run_configs['align'] is False:
        print(f"WARNING you are running {run_configs['align']} so no alignment will be conducted.")
    for idx, ds_row in enumerate(tqdm(truncated_ds)):
        if args.recover and (idx <= len(samples) -1):
            print(f"SKIPPING {idx}")
            continue

        print(f"ds_row:")
        print(f"{ds_row}")
        current_prompt = ds_row
        start = time.time()

        res, tokens, scores = runprompt(search, current_prompt, float(run_configs["rm_weight"]),
                                        run_configs["topk"], args.max_new_token,
                                        run_configs["mode"], run_configs["sample_temp"],
                                        llm_dev=args.llm_gpu, debug=run_configs['debug'], align=run_configs['align'])
        score_overall.append(scores)
        if tokens is None:
            print("Too long, skipped")
            continue

        elapsed = time.time() - start
        data[idx] = {"prompt": current_prompt, "response": res,
                     "elapsed":elapsed, "method": args.out_file + f"_{config_num}"}
        print(f"[DEBUG]: {elapsed=} {len(current_prompt)=} {current_prompt=}, {res=}")
        # saving periodically in case we lose progress:
        if idx % 10 == 0:
            print(f"saving outputs to {json_save_path}")
            start = time.time()
            with open(Path(json_save_path), "w") as outfile:
                json.dump(data, outfile, indent=4, ensure_ascii=False)
            elapsed = time.time() - start
            print(f"json save time: {elapsed:.3f}")
    with open(Path(json_save_path), "w") as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)
        print(f"run output saved to {json_save_path}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--run_percent", type=float, default=5.)
    parser.add_argument("--max_new_token", type=int, default=128) # 128

    parser.add_argument("--llm_gpu", type=str, default="cuda:0")
    parser.add_argument("--rm_gpu", type=str, default="cuda:1")
    parser.add_argument("--rm2_gpu", type=str, default="cuda:1")
    parser.add_argument("--recover", action='store_true', default=False)

    parser.add_argument("--config", type=str, default="configs/direct_config.yaml")
    parser.add_argument("--task_type", default="direct", type=str)

    # parser.add_argument("--config", type=str, default="configs/indirect_config.yaml")
    # parser.add_argument("--task_type", default="indirect", type=str)

    # parser.add_argument("--config", type=str, default="configs/indirect_collab_config.yaml")
    # parser.add_argument("--task_type", default="collab", type=str)

    parser.add_argument("--out_file", type=str, default="run_outs/llama")

    args = parser.parse_args()

    print(f"{args=}")
    # HF_HOME=/fs/nexus-scratch/jianyu34/.cache/huggingface/
    # hf_cache = os.path.join(hf_home, "hub")

    # for i in range(torch.cuda.device_count()):
    #     print(torch.cuda.get_device_properties(i).name)
    main(args)