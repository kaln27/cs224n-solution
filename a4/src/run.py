import random
import argparse

import dataset
import models
import trainer
import utils

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)

def pretrain(model, pretrain_dataset, args):
    assert args.writing_params_path is not None
    # TODO [part f]:
    # - Given:
    #     1. A corpus specified in args.pretrain_corpus_path
    #     2. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. Pretrain the model on this corpus
    #     2. Save the resulting model in args.writing_params_path

    # - Make sure to use the following hyperparameters for pretraining:
    # Hyperparameters for pretraining:
    # max_epochs=650
    # batch_size=128
    # learning_rate=args.pretrain_lr
    # lr_decay=True
    # warmup_tokens=512*20
    # final_tokens=650*len(pretrain_dataset)*block_size
    # num_workers=4
    # writer=writer

    ### YOUR CODE HERE ###
    tconf = trainer.TrainerConfig(max_epochs=650, batch_size=128, learning_rate=args.finetune_lr,
                    lr_decay=True, warmup_tokens=512*20, final_tokens=650*len(pretrain_dataset)*args.block_size,
                    num_workers=4, writer=args.writer, ckpt_path=args.writing_params_path, device=args.device)
    trainer_ = trainer.Trainer(model, pretrain_dataset, None, tconf)
    trainer_.train()
    trainer_.save_checkpoint()
    ### END YOUR CODE ###

def finetune(model, pretrain_dataset, args):
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None
    # TODO [part c] [part f]:
    # - Given:
    #     1. A finetuning corpus specified in args.finetune_corpus_path
    #     2. A path args.reading_params_path containing pretrained model
    #         parameters, or None if finetuning without a pretrained model
    #     3. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. If args.reading_params_path is specified, load these parameters
    #         into the model
    #     2. Finetune the model on this corpus
    #     3. Save the resulting model in args.writing_params_path
    # - Make sure to use the following hyperparameters:
    #     [part d] Hyperparameters for finetuning WITHOUT a pretrained model:
    #         max_epochs=75
    #         batch_size=256
    #         learning_rate=args.finetune_lr
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #         writer=writer
    #     [part f] Hyperparameters for finetuning WITH a pretrained model:
    #         max_epochs=10
    #         batch_size=256
    #         learning_rate=args.finetune_lr
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #         writer=writer
    #     You can use the args.reading_params_path flag to switch between the
    #     number of epochs for each case.

    ### YOUR CODE HERE ###
    if args.reading_params_path:
        state_dict = torch.load(args.reading_params_path, weights_only=False)
        model.load_state_dict(state_dict)

    finetune_data = open(args.finetune_corpus_path, encoding='utf-8').read()
    finetune_dataset = dataset.NameDataset(pretrain_dataset, finetune_data)
    tconf = trainer.TrainerConfig(max_epochs=10, batch_size=256, learning_rate=args.finetune_lr,
                    lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(pretrain_dataset)*args.block_size,
                    num_workers=4, writer=args.writer, ckpt_path=args.writing_params_path, device=args.device)
    trainer_ = trainer.Trainer(model, finetune_dataset, None, tconf)
    trainer_.train()
    trainer_.save_checkpoint()
    ### END YOUR CODE ###

def evaluate(model, pretrain_dataset, args):
    if args.device != 0: return
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    model.load_state_dict(torch.load(args.reading_params_path, weights_only=False))
    correct = 0
    total = 0
    with open(args.outputs_path, 'w', encoding='utf-8') as fout:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path, encoding='utf-8')):
            x = line.split('\t')[0]
            x = x + '⁇'
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x],
                            dtype=torch.long)[None,...].to(args.device)
            pred = utils.sample(model, x, 32, sample=False)[0]
            completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = completion.split('⁇')[1]
            predictions.append(pred)
            fout.write(pred + '\n')
        total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        print(f'Correct: {correct} out of {total}: {correct/total*100}%')
    else:
        print(f'Predictions written to {args.outputs_path}; no targets provided')

def task(rank, args):
    ddp_setup(rank, args.world_size)
    device = rank

    writer = SummaryWriter(log_dir='expt/%s/%s_%s_pt_lr_%f_ft_lr_%f' % (
        args.function,
        args.tb_expt_name,
        args.variant,
        args.pretrain_lr,
        args.finetune_lr))

    # Keep the block size 128
    # Why is the pretraining corpus always required (even if we're not pretraining?)
    # It's because we're using it as a hack to always have the same vocabulary
    # (that is, the same mapping from character to integer, and we build the
    # vocab from the pretraining corpus.)
    block_size = 128
    text = open(args.pretrain_corpus_path, encoding='utf-8').read()
    pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

    args.block_size = block_size
    args.device = device
    args.writer = writer

    # We don't suggest you change these hyperparameters, as they're known to work.
    # use them for both the vanilla and the RoPE models
    mconf = models.GPTConfig(
        pretrain_dataset.vocab_size,
        pretrain_dataset.block_size,
        n_layer=4,
        n_head=8,
        n_embd=256)

    # define models.
    # note: models should moved to device defined on lines 30-34.

    model = None
    if args.variant == 'vanilla':
        # TODO: [part c] Make some model here
        ### YOUR CODE HERE ###
        model = models.GPT(mconf).to(device)
        ### END YOUR CODE ###
    elif args.variant == 'rope':
        # TODO: [part g] Make some other model here
        # set mconf.rope parameter
        ### YOUR CODE HERE ###
        mconf.rope = True
        model = models.GPT(mconf).to(device)
        ### END YOUR CODE ###
    else:
        raise ValueError("Unknown model variant")

    print('Model on device: ', next(model.parameters()).device)

    # Perform pretraining, finetuning, or evaluation
    if args.function == 'pretrain':
        pretrain(model, pretrain_dataset, args)
    elif args.function == 'finetune':
        finetune(model, pretrain_dataset, args)
    elif args.function == 'evaluate':
        evaluate(model, pretrain_dataset, args)
    
    destroy_process_group()

if __name__ == '__main__':
    random.seed(0)

    argp = argparse.ArgumentParser()
    argp.add_argument('function', help="Choose pretrain, finetune, or evaluate")
    argp.add_argument('variant', help="Choose vanilla or rope")
    argp.add_argument('pretrain_corpus_path', default=None)
    argp.add_argument('--reading_params_path',default=None)
    argp.add_argument('--writing_params_path',default=None)
    argp.add_argument('--finetune_corpus_path', default=None)
    argp.add_argument('--eval_corpus_path', default=None)
    argp.add_argument('--outputs_path', default=None)
    argp.add_argument('--pretrain_lr', default=6e-3, type=float)
    argp.add_argument('--finetune_lr', default=6e-4, type=float)
    argp.add_argument('--tb_expt_name', help='debug string for tb log.',
                    default='run')
    args = argp.parse_args()
    
    args.world_size = torch.cuda.device_count()

    mp.spawn(task, args=(args,), nprocs=args.world_size)