"""
Data Processor for DeepSeek Educational Curriculum Model
Handles dataset loading, preprocessing, tokenization, and binary export.
"""

import os
from math import ceil
from typing import Dict, List

import numpy as np
import torch
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm


def load_encoder_decoder():
    """Return the same GPT-2 BPE encoder for encode/decode."""
    enc = tiktoken.get_encoding("gpt2")
    return enc, enc


class DeepSeekDataProcessor:
    def __init__(self, config=None):
        # GPT-2 BPE tokenizer
        self.enc = tiktoken.get_encoding("gpt2")

        # Special tokens (will be BPE-encoded; no added vocab)
        self.special_tokens = {
            "curriculum_start": "<|curriculum|>",
            "curriculum_end": "</|curriculum|>",
            "prompt_start": "<|prompt|>",
            "prompt_end": "</|prompt|>",
        }

        # Data dir
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Data directory: {self.data_dir}")

        # Processing config
        self.max_length = 4096
        self.min_length = 50  # characters

    # ------------ basic text utils ------------ #

    def preprocess_text(self, text: str) -> str:
        text = (text or "").lower()
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        return text

    def extract_curriculum_elements(self, example: Dict) -> Dict:
        prompt = self.preprocess_text(example.get("prompt", ""))
        curriculum = self.preprocess_text(example.get("text", ""))
        return {"prompt": prompt, "curriculum": curriculum}

    # ------------ encode/decode helpers ------------ #

    def decode_tokens(self, token_ids: List[int]) -> str:
        try:
            return self.enc.decode(token_ids)
        except Exception as e:
            print(f"Error decoding tokens: {e}")
            return ""

    def encode_text(self, text: str) -> List[int]:
        try:
            return self.enc.encode_ordinary(text)
        except Exception as e:
            print(f"Error encoding text: {e}")
            return []

    # ------------ map/process per example ------------ #

    def process(self, example: Dict) -> Dict:
        el = self.extract_curriculum_elements(example)

        if not el["curriculum"] or not el["prompt"]:
            return {"ids": [], "len": 0}

        full_text = (
            f"{self.special_tokens['prompt_start']} {el['prompt']} {self.special_tokens['prompt_end']} "
            f"{self.special_tokens['curriculum_start']} {el['curriculum']} {self.special_tokens['curriculum_end']}"
        )

        try:
            ids = self.enc.encode_ordinary(full_text)
            if len(ids) > self.max_length:
                ids = ids[: self.max_length]
            if len(ids) < 20:
                return {"ids": [], "len": 0}
            return {"ids": ids, "len": len(ids)}
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            return {"ids": [], "len": 0}

    # ------------ binary export & dataset prep ------------ #

    def _write_split_to_bin(self, split_name: str, tokenized, filename: str):
        """
        Robust writer:
        - Adapts shard count to dataset size/volume
        - Skips empty shards
        - Uses uint16 (GPT-2 vocab <= 50257 fits)
        """
        print(f"Saving {split_name} split to: {filename}")

        # Sum total token length
        arr_len = np.sum(tokenized["len"], dtype=np.uint64)
        if arr_len == 0:
            raise RuntimeError(f"No tokens to write for split '{split_name}'")

        dtype = np.uint16
        vocab_ceiling = 65535
        # Optional safety check on a small sample
        sample_ids = tokenized[0]["ids"] if len(tokenized) > 0 else []
        if any(i > vocab_ceiling for i in sample_ids):
            raise ValueError(
                "Found token id > 65535; uint16 not sufficient. Use uint32."
            )

        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        desired_max_shards = 1024
        n_rows = len(tokenized)

        # Prefer sharding by token volume for balance
        target_tokens_per_shard = 1_000_000
        est_shards_by_volume = max(1, int(ceil(int(arr_len) / target_tokens_per_shard)))
        num_shards = max(1, min(desired_max_shards, n_rows, est_shards_by_volume))

        idx = 0
        for batch_idx in tqdm(range(num_shards), desc=f"writing {filename}"):
            batch = (
                tokenized.shard(
                    num_shards=num_shards, index=batch_idx, contiguous=True
                ).with_format("numpy")
            )
            if len(batch) == 0:
                continue

            ids_lists = [ids for ids in batch["ids"] if len(ids) > 0]
            if not ids_lists:
                continue

            arr_batch = np.concatenate(ids_lists)
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()

        if idx != arr_len:
            print(
                f"[warn] wrote {idx} tokens but expected {arr_len}; file remains readable"
            )

        if os.path.exists(filename):
            print(f"Successfully created {filename}")
            print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
        else:
            raise RuntimeError(f"Failed to create {filename}")

    def prepare_dataset(self) -> Dict:
        """Load, filter, split, tokenize, and write .bin files."""
        print("Loading Education-Researchers dataset...")
        ds = load_dataset("ajibawa-2023/Education-Researchers")

        train_bin_path = os.path.join(self.data_dir, "train.bin")
        val_bin_path = os.path.join(self.data_dir, "validation.bin")
        finetune_bin_path = os.path.join(self.data_dir, "finetune.bin")

        print("Checking for existing processed files...")
        if (
            os.path.exists(train_bin_path)
            and os.path.exists(val_bin_path)
            and os.path.exists(finetune_bin_path)
        ):
            print("Found existing processed files!")
            print(
                f"Train: {os.path.getsize(train_bin_path)/(1024*1024):.2f} MB | "
                f"Val: {os.path.getsize(val_bin_path)/(1024*1024):.2f} MB | "
                f"Finetune: {os.path.getsize(finetune_bin_path)/(1024*1024):.2f} MB"
            )
            return {
                "train": train_bin_path,
                "validation": val_bin_path,
                "finetune": finetune_bin_path,
            }

        print("Processing dataset...")

        # Filter by character length
        def filter_by_length(example):
            text_length = len(example.get("text", "") or "")
            return self.min_length <= text_length <= 2000

        ds = ds.filter(filter_by_length)
        print(f"After filtering: {len(ds['train'])} examples")

        # Split: 80% train, 10% val, 10% finetune
        train_val = ds["train"].train_test_split(test_size=0.2, seed=42)
        val_finetune = train_val["test"].train_test_split(test_size=0.5, seed=42)
        ds_splits = {
            "train": train_val["train"],
            "validation": val_finetune["train"],
            "finetune": val_finetune["test"],
        }

        print("Dataset split sizes:")
        for k, v in ds_splits.items():
            print(f"- {k}: {len(v)}")

        # Process & write each split
        for split_name, split_data in ds_splits.items():
            print(f"\nProcessing {split_name} split...")

            # Safer remove_columns
            removable = [
                c for c in ["text", "prompt", "text_token_length"] if c in split_data.column_names
            ]

            tokenized = split_data.map(
                self.process,
                remove_columns=removable,
                desc=f"tokenizing {split_name} split",
                num_proc=8,
            )

            tokenized = tokenized.filter(lambda x: x["len"] > 0)
            print(f"After processing: {len(tokenized)} valid examples")

            outfile = {
                "train": train_bin_path,
                "validation": val_bin_path,
                "finetune": finetune_bin_path,
            }[split_name]

            self._write_split_to_bin(split_name, tokenized, outfile)

        return {
            "train": train_bin_path,
            "validation": val_bin_path,
            "finetune": finetune_bin_path,
        }

    # ------------ training helpers ------------ #

    def load_binary_data(self, filepath: str) -> torch.Tensor:
        try:
            data = np.memmap(filepath, dtype=np.uint16, mode="r")
            return torch.from_numpy(data.copy())
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            raise

    def get_batch(self, data: torch.Tensor, batch_size: int, block_size: int):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size].long() for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size].long() for i in ix])
        return x, y


def main():
    print("DeepSeek Education Curriculum Data Processor")
    print("=" * 50)
    processor = DeepSeekDataProcessor()
    processor.prepare_dataset()
    print("\nData processing completed successfully!")
    print("Files created:")
    print("- src/data/train.bin")
    print("- src/data/validation.bin")
    print("- src/data/finetune.bin")


if __name__ == "__main__":
    main()
