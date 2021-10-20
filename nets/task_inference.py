import torch
from torch import nn
from transformers.models.bart import BartModel
from transformers import AutoModel
from torch.nn import functional as F

class LongTermState:
    task_emb = None
    n = 0

    def __init__(self, config, trainable=False, weight=None, idx=None):
        self.trainable = trainable
        self.weight = weight
        self.idx = idx

    def reset_state(self):
        self.task_emb = None
        self.n = 0

    def update_emb(self, emb):
        if self.n == 0:
            self.task_emb = emb
        else:
            self.task_emb = (self.n / (self.n + 1)) * self.task_emb + (1 / (self.n + 1)) * emb
        self.n += 1

    def update_emb_batch(self, embs):
        for b in range(embs.size(0)):
            self.update_emb(embs[b])

    def get_task_emb(self):
        if not self.trainable:
            return self.task_emb.detach()
        else:
            return self.weight[self.idx]


class ShortTermState:
    def __init__(self, config):
        self.stm_size = config.stm_size
        self.stm = []
        self.reset_state()

    def reset_state(self):
        self.stm = []

    def update_emb(self, emb):
        self.stm.append(emb)
        if len(self.stm) > self.stm_size:
            self.stm = self.stm[1:]

    def update_emb_batch(self, embs):
        for b in range(embs.size(0)):
            self.update_emb(embs[b])

    def get_task_emb(self):
        emb = torch.mean(torch.stack(self.stm), 0)
        return emb.detach()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class SingleTaskEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.task_encoder_model == 'bart':
            self.bart = BartModel.from_pretrained('facebook/bart-base')
        else:
            self.bart = AutoModel.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1') # actually works better

    def forward(self, cq_input, cq_input_mask, ans_input, ans_input_mask):
        if self.config.task_encoder_model == 'bart':
            outputs = self.bart(cq_input, cq_input_mask, ans_input, ans_input_mask)
            ret = outputs.last_hidden_state.sum(1)
        else:
            outputs = self.bart(cq_input, cq_input_mask)
            ret = mean_pooling(outputs, cq_input_mask)
        return ret
