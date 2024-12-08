# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from functools import partial

from ft_datasets import (
    get_pure_bad_dataset,
    get_samsum_dataset,
    get_sqlgen_dataset,
)
from typing import Optional


DATASET_PREPROC = {
    "pure_bad_dataset": partial(get_pure_bad_dataset, max_words=512),
    "samsum_dataset": partial(get_samsum_dataset, max_words=512),
    "sqlgen_dataset": partial(get_sqlgen_dataset, max_words=512),
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    print(dataset_config)
    print(dataset_config.dataset)
    print(dataset_config.train_split)

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )

    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )
