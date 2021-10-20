import torch
from torch import nn
import random as _random

random = _random.Random(0)

def label_smoothed_nll_loss(lprobs, target, epsilon=0.1, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def total_param_dim(list_of_sizes):
    d = 0
    for sizes in list_of_sizes:
        m = 1
        for x in sizes:
            m *= x
        d += m
    return d


class TaskEmbMemory(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mem_limit = 10000
        self.register_buffer('x', torch.zeros(self.mem_limit, args.task_emb_dim))
        self.register_buffer('task_ids', torch.zeros(self.mem_limit).long())
        self.seen = 0

    def store(self, task_emb, task_id):
        if self.seen < self.mem_limit:
            j = self.seen
        else:
            j = random.randint(0, self.seen)
        if j < self.mem_limit:
            self.x[j] = task_emb
            self.task_ids[j] = task_id
        self.seen += 1

    def sample(self):
        if self.seen == 0:
            return None, None
        idx = random.randint(0, min(self.mem_limit, self.seen) - 1)
        return self.x[idx], self.task_ids[idx].item()
