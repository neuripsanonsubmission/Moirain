import torch
from torch import nn
import torch.nn.functional as F

from models.primitives import Embedding, RMSNorm, Linear
from models.attention import GPTAttentionFast, SelfAttention, GeometricAttention, CrossAttentionFast


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
    


class EmbedderAA(nn.Module):

    def __init__(self, model_conf, tokenizer):

        super().__init__()

        self.tokenizer = tokenizer
        self.types_num_aa = self.tokenizer.toktypes_num
        self.model_conf = model_conf

        self.bins_plddt = 32

        pad_aa = self.tokenizer.pad_idx

        self.type_embedder_aa = Embedding(self.types_num_aa, self.model_conf.c_s, padding_idx = pad_aa, init='bietti')
        self.plddt_embedder_aa = Embedding(self.bins_plddt, self.model_conf.c_s, init='bietti')

        self.layer_norm = RMSNorm(self.model_conf.c_s)

    def bin(self, x):

        x = x.clamp(0.0, 1.0)

        bin_idx = torch.floor(x * self.bins_plddt).long()
        bin_idx = torch.clamp(bin_idx, max=self.bins_plddt - 1)

        return bin_idx

        
    def forward(self, batch):
        
        s_na_embed = self.layer_norm(self.type_embedder_aa(batch['ttype_aa']) + self.plddt_embedder_aa(self.bin(batch['plddt_aa'])))

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
    

class EncoderAA(nn.Module):

    def __init__(self, model_conf):
        super().__init__()

        self.c_s = model_conf.c_s
        self.n = model_conf.transition_n

        self.self_mha = SelfAttention(model_conf)

        self.geom_mha = GeometricAttention(model_conf)

        self.feed_forward = nn.Sequential(
            Linear(self.c_s, self.n * self.c_s, init='relu'),
            nn.GELU(),
            Linear(self.n * self.c_s, self.c_s, init='glorot')
        )

        self.layer_norm_1 = RMSNorm(self.c_s)
        self.layer_norm_2 = RMSNorm(self.c_s)
        self.layer_norm_3 = RMSNorm(self.c_s)


    def forward(self, s, rpos, T, pad):
        
        s = s + self.self_mha(s, rpos, pad)

        s = self.layer_norm_1(s)

        s = s + self.geom_mha(s, s, T, T, pad, pad)

        s = self.layer_norm_2(s)

        s = s + self.feed_forward(s)

        s = self.layer_norm_3(s)

        return s
    


class EncoderCross(nn.Module):

    def __init__(self, model_conf):
        super().__init__()

        self.c_s1 = model_conf.c_s1
        self.n = model_conf.transition_n

        self.cross_mha = CrossAttentionFast(model_conf)

        self.layer_norm_1 = RMSNorm(self.c_s1)


    def forward(self, s1, s2, pad1, pad2):
        
        s1 = s1 + self.cross_mha(s1, s2, pad1, pad2)

        s1 = self.layer_norm_1(s1)

        return s1




class MainModel(nn.Module):

    def __init__(self, model_conf, tokenizer):
        super().__init__()

        self.model_conf = model_conf

        self.type_embedder = Embedder(self.model_conf.na, tokenizer.na)
        self.type_aa_embedder = EmbedderAA(self.model_conf.aa, tokenizer.aa)

        self.trunk = nn.ModuleDict()

        for b in range(self.model_conf.aa.num_blocks_aa):
            self.trunk[f'encoder_aa_{b}'] = EncoderAA(self.model_conf.aa)

        self.cross_indices = set(round(i * (self.model_conf.na.num_blocks_na - 1) / (self.model_conf.cross.num_blocks - 1))for i in range(self.model_conf.cross.num_blocks))

        for b in range(self.model_conf.na.num_blocks_na):
            self.trunk[f'encoder_na_{b}'] = EncoderNA(self.model_conf.na)
            if b in self.cross_indices:
                self.trunk[f'encoder_cross_{b}'] = EncoderCross(self.model_conf.cross)

        self.log_head_type = nn.Linear(self.model_conf.na.c_s, self.model_conf.na.c_lm_head)

        
    def forward(self, batch):
        
        s_aa = self.type_aa_embedder(batch)
        s_na = self.type_embedder(batch)

        for b in range(self.model_conf.aa.num_blocks_aa):
            s_aa = self.trunk[f'encoder_aa_{b}'](s_aa, batch['tpos_aa'], batch['T_aa'], batch['pad_aa'])

        for b in range(self.model_conf.na.num_blocks_na):
            s_na = self.trunk[f'encoder_na_{b}'](s_na, batch['tpos_na'])
            if b in self.cross_indices:
                s_na = self.trunk[f'encoder_cross_{b}'](s_na, s_aa, batch['pad_na'], batch['pad_aa'])
        
        logits_type_na = self.log_head_type(s_na)
        log_probs_type_na = F.log_softmax(logits_type_na, dim=-1)

        model_out = {
            'logits_type_na': logits_type_na,
            'log_probs_type_na': log_probs_type_na
        }

        return model_out
    

    def forward_aa(self, batch):
        
        s_aa = self.type_aa_embedder(batch)

        for b in range(self.model_conf.aa.num_blocks_aa):
            s_aa = self.trunk[f'encoder_aa_{b}'](s_aa, batch['tpos_aa'], batch['T_aa'], batch['pad_aa'])

        return s_aa
    

    def forward_na(self, batch, s_aa):
        
        s_na = self.type_embedder(batch)

        for b in range(self.model_conf.na.num_blocks_na):
            s_na = self.trunk[f'encoder_na_{b}'](s_na, batch['tpos_na'])
            if b in self.cross_indices:
                s_na = self.trunk[f'encoder_cross_{b}'](s_na, s_aa, batch['pad_na'], batch['pad_aa'])
        
        logits_type_na = self.log_head_type(s_na)
        log_probs_type_na = F.log_softmax(logits_type_na, dim=-1)

        model_out = {
            'logits_type_na': logits_type_na,
            'log_probs_type_na': log_probs_type_na
        }

        return model_out