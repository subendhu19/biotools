## Adapted from https://github.com/moskomule/ewc.pytorch/blob/master/utils.py
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import logging
logger = logging.getLogger(__name__)

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

def get_diag_fisher(model,old_task_dataset,args):

    cl_epoch_iterator = iter(old_task_dataset)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if torch.cuda.is_available():
        devices=list(set([p.device for n, p in model.named_parameters() if p.requires_grad]))
        src_device=devices[0]
        if devices.__len__()>1:
            logger.warning('Model is in {0} devices. Behavior may be unstable for device>1'.format(devices))
    else:
        src_device=torch.device("cpu")
    means = {}
    for n, p in deepcopy(params).items():
        means[n] = variable(p.data).to(src_device)
    precision_matrices = {}
    for n, p in deepcopy(params).items():
        p.data.zero_()
        precision_matrices[n] = variable(p.data).to(src_device)

    model.eval()
    for step_idx,batch in enumerate(cl_epoch_iterator):
        cl_inputs, cl_labels = mask_tokens(cl_batch, tokenizer, args) if args.mlm else (cl_batch, cl_batch)
        cl_inputs = cl_inputs.to(args.device)
        cl_labels = cl_labels.to(args.device)
        model.zero_grad()
        input = variable(input).to(src_device)

        output = self.model(input).view(1, -1)
        label = output.max(1)[1].view(-1)
        loss = F.nll_loss(F.log_softmax(output, dim=1), label)
        loss.backward()

        for n, p in self.model.named_parameters():
            precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

    precision_matrices = {n: p for n, p in precision_matrices.items()}
    return precision_matrices

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target) + importance * ewc.penalty(model)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)
