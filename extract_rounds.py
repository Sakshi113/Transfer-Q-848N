from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt

def get_response(database, idx):
    idx = str(idx)
    line = database[idx]['response']
    if type(line) == dict:
        return len(line), database[idx]['elapsed']
    else:
        return 1, database[idx]['elapsed']


def main(file1, num_eval):


    with open(file1, "r") as file:
        db1 = json.load(file)

    key_list = list(db1.keys())[1:]
    num_lines = []
    eval_times = []
    for idx in tqdm(key_list):
        prompt = db1[idx]["prompt"]

        num_rounds, time_spent = get_response(db1, idx)
        num_lines.append(num_rounds)
        eval_times.append(time_spent)
        if int(idx) >= num_eval:
            break
    print(file1)
    print(f"for {len(num_lines)} entries, averaged {np.average(num_lines)} rounds of evaluation, {np.average(eval_times)} seconds")

    # We can set the number of bins with the *bins* keyword argument.
    plt.hist(np.array(num_lines), bins=5)
    plt.show()


    plt.show()
if __name__ == "__main__":
    """
    only pulling number of rounds for each conversation:
    """
    num_eval = 48

    # file1 = "run_outs/safenlp_llm_only_direct_0.jsonl"
    # file1 = "run_outs/safenlp_indirect_0.jsonl"
    # file1 = "run_outs/safenlp_no_align_critic_worker_collab_0.jsonl"
    file1 = "run_outs/safenlp_collab_0.jsonl"


    main(file1, num_eval)
