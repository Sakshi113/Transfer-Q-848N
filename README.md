# Transfer Q*: Principled Decoding for LLM Alignment

This codebase provides a Pytorch implementation for the paper: Transfer Q Star: Principled Tuning-free Decoding for LLM Alignment

## Setup
The packages and versions used are mentioned in requirements.txt
```
conda create -n tq python=3.9 -y
conda activate tq

cd transfer_q
mkdir run_outs
pip install -r requirements.txt
```

# For direct transfer tasks on HH-RLHF dataset run the following command:

```
python collect_model_outs.py --config="configs/direct_config.yaml" --task_type="direct" --dataset="Dahoas/full-hh-rlhf"
```

# For indirect transfer tasks on HH-RLHF run the following command:

```
python collect_model_outs.py --config="configs/indirect_config.yaml" --task_type="indirect" --dataset="Dahoas/full-hh-rlhf"
```

# For collaborative indirect transfer task:

```
python collect_model_outs.py --config="configs/indirect_collab_config.yaml" --task_type="collab" --dataset="Dahoas/full-hh-rlhf"

```
# To measure reward of generated responses run the following command:

```
python measure_reward.py --out_file="run_outs/example_out_0.jsonl"
```

## References

The codebase has been adapted from [TransferQ](https://github.com/Soumya1612-Rasha/Transfer-Q).

## For bibtex citation 

```
@misc{TBD,
      title={...}, 
      author={Souradip Chakraborty and Soumya Suvra Ghosal and Ming Yin and Dinesh Manocha and Mengdi Wang and Amrit Singh Bedi and Furong Huang},
      year={2026},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={...}, 
}
```
