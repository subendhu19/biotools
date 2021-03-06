# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
import copy
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn.parallel import data_parallel

from copy import deepcopy
from torch.autograd import Variable
from custom_data_parallel import custom_data_parallel, CustomDataParallel

GRAD_CLIP_VALUE_EWC=10

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def get_l2_norm_materials(model,args):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad and "LayerNorm" not in n}
    means = {}
    precision_matrices = {}
    for n, p in params.items():
        means[n] = p.clone().detach()
        precision_matrices[n]= None

    return means,precision_matrices

def get_diag_fisher(model,old_task_dataset_len,old_task_dataset_iterator,args,tokenizer,chunk_sizes):
    logger.info('Computing Diagonal Fisher estimates for EWC ...')
    logger.info('EWC computation chunks sizes {0}'.format(chunk_sizes))
    #cl_epoch_iterator = iter(old_task_dataset_iterator)
    cl_epoch_iterator = tqdm(old_task_dataset_iterator, desc="Iteration")
    params = {n: p for n, p in model.named_parameters() if p.requires_grad and "LayerNorm" not in n}
    if torch.cuda.is_available():
        devices=list(set([p.device for n, p in model.named_parameters() if p.requires_grad]))
        src_device=devices[0]
        if devices.__len__()>1:
            logger.warning('Model is in {0} devices. Behavior may be unstable for device>1'.format(devices))
    else:
        src_device= -1
    means = {}
    precision_matrices = {}
    for n, p in params.items():
        means[n] = p.clone().detach()
        precision_matrices[n] = torch.zeros_like(p,dtype=p.dtype,device=p.device)
    model.eval()
    for step_idx,batch in enumerate(cl_epoch_iterator):
        model.zero_grad()
        cl_inputs, cl_labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        cl_inputs = cl_inputs.to(args.device)
        cl_labels = cl_labels.to(args.device)
        if src_device==-1:
            outputs = model(cl_inputs,masked_lm_labels=cl_labels)
        else:
            #if len(devices)==1:
            #    outputs=model(cl_inputs,masked_lm_labels=cl_labels)
            #else:
            outputs = custom_data_parallel(model,cl_inputs,module_kwargs={'masked_lm_labels': cl_labels}, device_ids=None, output_device=src_device,chunk_sizes=chunk_sizes)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        if args.n_gpu > 1:
            loss = loss.mean()
        loss.backward()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None and n in precision_matrices:
                    ## Clamping not mentioned in any EWC code or paper, but pretty sure this will be an issue in deep NLP archs.
                    precision_matrices[n] += \
                        torch.clamp(p.grad.data,-GRAD_CLIP_VALUE_EWC,GRAD_CLIP_VALUE_EWC) ** 2 / old_task_dataset_len
    precision_matrices = {n: p for n, p in precision_matrices.items()}
    return means,precision_matrices


def penalty(model, means, precision_matrices, ewc_type):
    loss = 0
    for n, p in model.named_parameters():
        if n.startswith('module.'):
            n = n.replace('module.', '')
        if n in precision_matrices:
            _loss = (p - means[n]) ** 2
            if ewc_type == 0:
                relevant_fisher = precision_matrices[n]
                _loss = _loss * relevant_fisher
            loss += _loss.sum()
    return loss

def load_and_cache_examples(args, tokenizer, evaluate=False,cl=False):
    if cl:
        file_path = args.cl_eval_data_file if evaluate else args.cl_train_data_file
    else:
        file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def get_distill_loss(current_label_outs, base_label_outs, labels):
    s = torch.nn.Softmax(dim=2)
    ls = torch.nn.LogSoftmax(dim=2)
    loss = (-1 * s(base_label_outs) * ls(current_label_outs + 1e-30)).sum(axis=2) * (labels > 0).float()
    return torch.mean(loss)

def project2cone(gradient,memories):
    # original code can be obtained from
    # https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
    # The above implementation required a quadprog solver which runs on cpu.
    # This would be too slow for our setup (moving grads gpu->cpu, quadprog(), and then cpu->gpu)
    # For only one memory gradient, we just use direct projection.
    # GEM's dual quad program requires projection onto a convex cone.
    # If we only have one previous memory gradient, there is no cone and this can be solved analytically
    # instead of running quadprog.
    if memories.numel() != gradient.numel():
        assert False,"specialized for only one memory gradient "
    gradient=gradient.squeeze()
    memories=memories.squeeze()
    unnormalized_projection = torch.dot(gradient,memories)
    if (unnormalized_projection <0).sum()==0:
        #print('UP: {0}, not violated'.format(unnormalized_projection))
        return gradient,False
    else:
        #print('UP: {0}, violated'.format(unnormalized_projection))
        constrained_vector = gradient - (unnormalized_projection/ torch.norm(memories,p=2)) * memories

        return constrained_vector, True



def obtain_grads(grads_memory,model):
    count=0
    for param in model.parameters():
        num_elements= param.numel()
        if param.grad is not None:
            grads_memory[count: count+num_elements].copy_(param.grad.data.view(-1))
        else:
            grads_memory[count: count+num_elements].fill_(0.)
        count+=num_elements

def overwrite_grads(model,grad_vec):
    count=0
    for param in model.parameters():
        num_elements= param.numel()
        if param.grad is not None:
            this_grad = grad_vec[count: count+num_elements].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count+=num_elements

def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
          cl_train_dataset=None, distil_model=None) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    train_chunk_sizes=[args.main_gpu_train_batch_size] + [args.per_gpu_train_batch_size]* max(0, args.n_gpu-1)

    args.train_batch_size = sum(train_chunk_sizes)
    cl_train_chunk_sizes = [args.cl_main_gpu_train_batch_size] + [args.cl_per_gpu_train_batch_size]\
                           * max(0, args.n_gpu-1)
    args.cl_train_batch_size = sum(cl_train_chunk_sizes)

    logger.info('Train chunk size distribution is {0}, total batch size is {1}'.format(train_chunk_sizes,args.train_batch_size))
    logger.info('CL Train chunk size distribution is {0}, total batch size is {1}'.format(cl_train_chunk_sizes,args.cl_train_batch_size))

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    # Dropping last batch because of unknown behavior from scatter using specific chunk sizes.
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate,drop_last=True
    )
    if cl_train_dataset and args.ewc_type !=1:
        cl_train_sampler = RandomSampler(cl_train_dataset) if args.local_rank == -1 else \
            DistributedSampler(cl_train_dataset)
        cl_batch_size= sum(cl_train_chunk_sizes)
        
        # Dropping last batch because of unknown behavior from scatter using specific chunk sizes.
        cl_train_dataloader = DataLoader(
            cl_train_dataset, sampler=cl_train_sampler, batch_size=cl_batch_size, collate_fn=collate,drop_last=True
        )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    ## Iterating over the old task dataset. Parallel_apply is used. Make sure to keep this before DataParallel call.
    if args.ewc is True:
        if args.ewc_type==0:
            assert cl_train_dataset is not None, "CL dataset needed for EWC"
            ewc_means, ewc_F = get_diag_fisher(model, old_task_dataset_len=len(cl_train_dataset),
                                               old_task_dataset_iterator=cl_train_dataloader, args=args,
                                               tokenizer=tokenizer,chunk_sizes=cl_train_chunk_sizes)

        elif args.ewc_type ==1:
            ewc_means, ewc_F = get_l2_norm_materials(model,args)
        else:
            assert False,"Only 0 and 1 EWC_type options supported"

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        #model = torch.nn.DataParallel(model)
        model = CustomDataParallel(model)
        if distil_model is not None:
            distil_model = CustomDataParallel(distil_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        if distil_model is not None:
            distil_model = torch.nn.parallel.DistributedDataParallel(
                distil_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
            )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Instantaneous batch size source GPU = %d", args.main_gpu_train_batch_size)

    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility

    if distil_model is not None:
        distil_model_to_resize = distil_model.module if hasattr(distil_model, "module") else distil_model
        distil_model_to_resize.eval()  # No gradients for the distil model
        distil_model_to_resize.resize_token_embeddings(len(tokenizer))
        distil_model.chunk_sizes = cl_train_chunk_sizes

    if args.gem is True:
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        grads_memory = torch.Tensor(sum(grad_dims),)
        current_task_grad = torch.Tensor(sum(grad_dims),)
        if torch.cuda.is_available():
            grads_memory = grads_memory.cuda()
            current_task_grad = current_task_grad.cuda()

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        if cl_train_dataset and args.ewc_type!=1:
            cl_epoch_iterator = iter(cl_train_dataloader)
        skipped_batches = 0
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            try:
                inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                # model.train()
                model.module.train() if hasattr(model, 'module') else model.train()
                model.chunk_sizes = train_chunk_sizes
                outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)


                # Rehearsal or Distillation mechanism
                #-----------------------------
                if cl_train_dataset and args.ewc is False:  # cl dataset provided and EWC is off. So rehersal is on.
                    try:
                        cl_batch = next(cl_epoch_iterator)
                    except StopIteration:
                        logger.info('Completed CL train batch iteration. Restarting ...')
                        cl_epoch_iterator = iter(cl_train_dataloader)
                        cl_batch = next(cl_epoch_iterator)

                    if distil_model is not None:
                        # Distillation mechanism
                        # -----------------------------
                        cl_inputs, cl_labels = mask_tokens(cl_batch, tokenizer, args) if args.mlm else (cl_batch, cl_batch)
                        cl_inputs = cl_inputs.to(args.device)
                        cl_labels = cl_labels.to(args.device)

                        model.chunk_sizes = cl_train_chunk_sizes
                        cl_outputs = model(cl_inputs, masked_lm_labels=cl_labels) if args.mlm else \
                            model(cl_inputs, labels=cl_labels)

                        with torch.no_grad():
                            distil_outputs = distil_model(cl_inputs, masked_lm_labels=cl_labels) if args.mlm else \
                                distil_model(cl_inputs, labels=cl_labels)

                        distil_loss = get_distill_loss(cl_outputs[1], distil_outputs[1], cl_labels)
                        loss += distil_loss * args.cl_loss_multiplier
                    else:
                        if args.gem is False:
                            # Rehearsal mechanism
                            # -----------------------------
                            cl_inputs, cl_labels = mask_tokens(cl_batch, tokenizer, args) if args.mlm else (cl_batch, cl_batch)
                            cl_inputs = cl_inputs.to(args.device)
                            cl_labels = cl_labels.to(args.device)
                            model.module.train() if hasattr(model, 'module') else model.train()
                            model.chunk_sizes = cl_train_chunk_sizes
                            cl_outputs = model(cl_inputs, masked_lm_labels=cl_labels) if args.mlm else model(cl_inputs,
                                                                                                             labels=cl_labels)
                            loss += cl_outputs[0] * args.cl_loss_multiplier
                        else:
                            # GRADIENT EPISODIC MEMORY.
                            # Rehearsal mechanism with gradient projection. Without explicit loss constraints.
                            # -----------------------------
                            cl_inputs, cl_labels = mask_tokens(cl_batch, tokenizer, args) \
                                if args.mlm else (cl_batch, cl_batch)
                            cl_inputs = cl_inputs.to(args.device)
                            cl_labels = cl_labels.to(args.device)
                            model.module.train() if hasattr(model, 'module') else model.train()
                            model.chunk_sizes = cl_train_chunk_sizes
                            cl_outputs = model(cl_inputs, masked_lm_labels=cl_labels) if args.mlm \
                                else model(cl_inputs,labels=cl_labels)
                            cl_loss=cl_outputs[0].mean()
                            cl_loss.backward()
                            obtain_grads(grads_memory,model)
                            model.module.zero_grad() if hasattr(model, 'module') else model.zero_grad()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logger.info('Skipping step {}'.format(step))
                    torch.cuda.empty_cache()
                    skipped_batches += 1
                    continue
                else:
                    raise e

            #-----------------------------
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            #EWC mechanism
            #-----------------------------
            if args.ewc:
                # import pdb;pdb.set_trace()
                ewc_penalty = penalty(model, ewc_means, ewc_F, ewc_type=args.ewc_type)
                if step % 50000 in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                    print(
                        'Showing loss components for few steps Loss: {0}, EWC_Loss {1}, effective EWC_Loss {2}'.format(
                            loss, ewc_penalty, args.cl_loss_multiplier * ewc_penalty))
                # print('Showing loss components for few steps Loss: {0}, EWC_Loss {1}, effective EWC_Loss {2}'.format(loss,ewc_penalty,args.cl_loss_multiplier*ewc_penalty))
                loss += float(args.cl_loss_multiplier) * ewc_penalty
            #-----------------------------

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            #GEM mechanism
            #-----------------------------
            if args.gem is True:
                obtain_grads(current_task_grad,model)
                projected_grad, violation = project2cone(current_task_grad,grads_memory)
                overwrite_grads(model, projected_grad)
            #-----------------------------

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    #torch.nn.utils.clip_grad_norm_(model.module.parameters() if hasattr(model, 'module')
                    #                              else model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        logger.info("Skipped total %d batches due to runtime errors", skipped_batches)
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_chunk_sizes = [args.main_gpu_eval_batch_size] + [args.per_gpu_eval_batch_size] * max(0, args.n_gpu - 1)

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = sum(eval_chunk_sizes)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate, drop_last=True
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    model.chunk_sizes=eval_chunk_sizes

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters

    ## Other parameters
    parser.add_argument("--ewc", action='store_true', help="Whether to run ewc CL training.")
    parser.add_argument("--gem", action='store_true', help="Whether to run gem CL training.")
    parser.add_argument("--ewc_type", default=0, type=int,
                        help="0:default EWC, 1: EWC with constant diagonal precision of 1 (L2) 2: Kronecker factorization")
    parser.add_argument("--num_ewc_steps", default=100, type=int,
                        help="Total number of steps to perform for estimating the EWC Laplace Approximation. Entire dataset is used if -1")
    parser.add_argument("--cl_train_data_file", default=None, type=str, required=False,
                        help="The input cl training data file (a text file).")

    parser.add_argument("--cl_eval_data_file", default=None, type=str,
                        help="An optional input evaluation cl data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--distil_model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--cl_per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per CL GPU/CPU for training.")

    parser.add_argument("--main_gpu_train_batch_size", default=1, type=int, help="Batch size of the source GPU for training.")
    parser.add_argument("--cl_main_gpu_train_batch_size", default=1, type=int,
                        help="Batch size the source CL GPU for training.")

    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--main_gpu_eval_batch_size", default=2, type=int, help="Batch size of the source GPU  for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--cl_loss_multiplier", default=0.1, type=float,
                        help="The loss multiplier")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    # ------ CL Configuration checks -------
    if args.ewc is False:
        if args.ewc_type==0:
            if args.distil_model_name_or_path is not None:
                if args.gem is True:
                    raise ValueError(" Why are you using GEM with distill model ?")
                else:
                    print('**** DISTILLATION CONTINUAL LEARNING MODE ******')
            else:
                if args.gem is True:
                    print('**** GRADIENT EPISODIC MEMORY CONTINUAL LEARNING MODE ******')
                else:
                    print('**** REHEARSAL CONTINUAL LEARNING MODE ******')
        elif args.ewc_type ==1:
            raise ValueError(" If ewc is off, ewc_type should point to the default value of 0")
        else:
            raise ValueError(" ewc_type value not supported")
    else:
        if args.gem is True:
            raise ValueError(" Why are you using GEM with EWC ?")
        if args.ewc_type==0:
            print('**** EWC CONTINUAL LEARNING MODE ******')
        elif args.ewc_type ==1:
            print('**** L2 CONTINUAL LEARNING MODE ******')
        else:
            raise ValueError(" ewc_type value not supported")



    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                    cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                    cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)

    if args.distil_model_name_or_path is not None:
        logger.info("Distillation model provided. Loading")
        distil_config = config_class.from_pretrained(args.distil_model_name_or_path, cache_dir=args.cache_dir)
        distil_model = model_class.from_pretrained(
            args.distil_model_name_or_path,
            from_tf=bool(".ckpt" in args.distil_model_name_or_path),
            config=distil_config,
            cache_dir=args.cache_dir,
        )
        distil_model.to(args.device)
    else:
        distil_model = None

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        if args.cl_train_data_file is not None and args.ewc_type!=1:
            cl_train_dataset=load_and_cache_examples(args, tokenizer, evaluate=False,cl=True)
        else:
            cl_train_dataset=None

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, cl_train_dataset=cl_train_dataset,
                                     distil_model=distil_model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
