# Evaluating Biases in LLM-Generated Musical Taste Profiles for Recommendation

One particularly promising use case of Large Language Models (LLMs) for recommendation is the automatic generation of Natural Language (NL) user taste profiles from consumption data. These profiles offer interpretable and editable alternatives to opaque collaborative filtering representations, enabling greater transparency and user control. However, it remains unclear whether users identify these profiles to be an accurate representation of their taste, which is crucial for trust and usability. Moreover, because LLMs inherit societal and data-driven biases, profile quality may systematically vary across user and item characteristics. In this paper, we study this issue in the context of music streaming, where personalization is challenged by a large and culturally diverse catalog. We conduct a user study in which participants rate NL profiles generated from their own listening histories. We analyze whether identification with the profiles is biased by user attributes (e.g., mainstreamness, taste diversity) and item features (e.g., genre, popularity, country of origin). We also compare these patterns to those observed when using the profiles in a downstream recommendation task. Our findings highlight both the potential and limitations of scrutable, LLM-based profiling in personalized systems.

## Dataset
We will release our proprietary data upon acceptance, ensuring anonymity.

## Quickstart

Build the docker image:

```sh
$ make build
```

Run a Docker container and start an interactive bash session, while mounting the current directory:
```sh
$ make run-bash
```

##
Paper plots

To generate the figures of the paper, refer to the notebook LLM_bias_plots.ipynb

##
Doubly Robust estimation

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
