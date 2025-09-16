# Download hugginface datasets into parquet.

import argparse
import datasets

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save parquet files.")
args = parser.parse_args()

train_dataset = datasets.load_dataset("jinulee-v/legit_ko_verl", split="train")
valid_dataset = datasets.load_dataset("jinulee-v/legit_ko_verl", split="valid")

train_dataset.to_parquet(f"{args.output_dir}/train.parquet")
valid_dataset.to_parquet(f"{args.output_dir}/valid.parquet")