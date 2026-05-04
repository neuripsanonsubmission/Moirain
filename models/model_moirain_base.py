import torch
from torch import nn
import torch.nn.functional as F

from models.primitives import Embedding, RMSNorm, Linear
from models.attention import GPTAttentionFast



class Embedder(nn.Module):

    def __init__(self, model_conf, tokenizer):

        super().__init__()

        self.tokenizer = tokenizer
        self.types_num_na = self.tokenizer.toktypes_num
        self.model_conf = model_conf

        pad_na = self.tokenizer.pad_idx

        self.type_embedder_na = Embedding(self.types_num_na, self.model_conf.c_s, padding_idx = pad_na, init='bietti')

        self.layer_norm = RMSNorm(self.model_conf.c_s)

        
    def forward(self, batch):
        
        s_na_embed = self.layer_norm(self.type_embedder_na(batch['ttype_na']))

        return s_na_embed

    

class EncoderNA(nn.Module):

    def __init__(self, model_conf):
        super().__init__()

        self.c_s = model_conf.c_s
        self.n = model_conf.transition_n

        self.self_mha = GPTAttentionFast(model_conf)

        self.feed_forward = nn.Sequential(
            Linear(self.c_s, self.n * self.c_s, init='relu'),
            nn.GELU(),
            Linear(self.n * self.c_s, self.c_s, init='glorot')
        )

        self.layer_norm_1 = RMSNorm(self.c_s)
        self.layer_norm_2 = RMSNorm(self.c_s)


    def forward(self, s1, rpos1):
        
        s1 = s1 + self.self_mha(s1, rpos1)

        s1 = self.layer_norm_1(s1)

        s1 = s1 + self.feed_forward(s1)

        s1 = self.layer_norm_2(s1)

        return s1




class MainModel(nn.Module):

    def __init__(self, model_conf, tokenizer):
        super().__init__()

        self.model_conf = model_conf

        self.type_embedder = Embedder(self.model_conf, tokenizer)

        self.trunk = nn.ModuleDict()

        for b in range(self.model_conf.num_blocks_na):
            self.trunk[f'encoder_na_{b}'] = EncoderNA(self.model_conf)

        self.log_head_type = nn.Linear(self.model_conf.c_s, self.model_conf.c_lm_head)

        
    def forward(self, batch):

        s_na = self.type_embedder(batch)

        for b in range(self.model_conf.num_blocks_na):
            s_na = self.trunk[f'encoder_na_{b}'](s_na, batch['tpos_na'])
        
        logits_type_na = self.log_head_type(s_na)
        log_probs_type_na = F.log_softmax(logits_type_na, dim=-1)

        model_out = {
            'logits_type_na': logits_type_na,
            'log_probs_type_na': log_probs_type_na
        }

        return model_out