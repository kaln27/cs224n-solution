#! /usr/bin/env python

# set up logging
import logging

# make deterministic
from mingpt.utils import set_seed

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from dataset import CharDataset
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import sample


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(42)

    block_size = 128 # spatial extent of the model for its context

    text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(text, block_size) # one line of poem is roughly 50 characters

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                    n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)



    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=2, batch_size=64, learning_rate=6e-4,
                        lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*block_size,
                        num_workers=8)
    trainer = Trainer(model, train_dataset, None, tconf)
    trainer.train()

    # alright, let's sample some character-level Shakespeare

    context = "O God, O God!"
    x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
    y = sample(model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]
    completion = ''.join([train_dataset.itos[int(i)] for i in y])
    print(completion)



if __name__ == '__main__':
    main()