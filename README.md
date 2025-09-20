# Biases in LLM-Generated Musical Taste Profiles for Recommendation

This repository provides our Python code to reproduce the experiments from the paper "Biases in LLM-Generated Musical Taste Profiles for Recommendation". Accepted to ACM recsys 2025. Link to the paper: https://arxiv.org/abs/2507.16708

Please cite our paper if you use this code in your own work:
```
@inproceedings{sguerra2025biases,
  title={Biases in LLM-Generated Musical Taste Profiles for Recommendation},
  author={Sguerra, Bruno and Epure, Elena V and Lee, Harin and Moussallam, Manuel},
  booktitle={Proceedings of the Nineteenth ACM Conference on Recommender Systems},
  pages={527--532},
  year={2025}
}
```
## Dataset
The `data` folder contains the following files:  
- **`user_data.csv`**: user profiles and ratings  
- **`long_term.csv`**: long-term preferences, used for computing the ATE with Doubly Robust  

## Quickstart

Build the docker image:

```sh
$ make build
```

Run a Docker container and start an interactive bash session, while mounting the current directory:
```sh
$ make run-bash
```

## Paper plots

To generate the figures of the paper, refer to the notebook LLM_bias_plots.ipynb.

## Doubly Robust estimation of ATE

The boostrapped estimations of ATE from the doubly robust method can be optained running doubly_robust.py in the srs folder.

## Downstream task

Download the fine-tuned cross-encoder, released by [**a previous work**](https://arxiv.org/abs/2411.05649):

```bash
wget https://zenodo.org/records/14289764/files/models.zip
apt-get update
apt-get install unzip
unzip models.zip -d models/ && rm models.zip
```

Train a new model on our dataset:
```bash
poetry run python -m  gpl.train  --path_to_generated_data "./data"    --base_ckpt "msmarco-bert-base-dot-v5"     --gpl_score_function "cos_sim"     --batch_size_gpl 10   --gpl_steps 10000   --output_dir "models/NL_profiles"   --cross_encoder "./models/cross-encoder-musiccaps-ms-marco-MiniLM-L-6-v2/"  --max_seq_length 512
```

Test the new model on our test datasets:
```bash
poetry run python src/eval.py --output_path results/ --input_path data/test/ --our_model_path models/NL_profiles/
```
