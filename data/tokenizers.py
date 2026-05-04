import numpy as np
from transformers import PreTrainedTokenizerFast


class Tokenizer:
    def __init__(self, vocab):
            
            self._toktypes = vocab

            self._toktypes_order = {toktype: i for i, toktype in enumerate(self._toktypes)}
            self._toktypes_num = len(self._toktypes)

    @property
    def toktypes(self):
        return self._toktypes
    
    @property
    def toktypes_order(self):
        return self._toktypes_order
    
    @property
    def toktypes_num(self):
        return self._toktypes_num

    @property
    def pad_idx(self):

        pad_idx = self.toktypes_order.get('<pad>', self.toktypes_num)

        if pad_idx == self.toktypes_num:
            raise ValueError(f"<pad> is not in the vocabulary")
        
        return pad_idx


    def encode(self, tokens):
        
        encodes = []

        for tok in tokens:
            
            toktype_idx = self.toktypes_order.get(tok, self.toktypes_num)

            if toktype_idx == self.toktypes_num:
                raise ValueError(f"Token '{tok}' is not in the vocabulary")

            encodes.append(toktype_idx)

        encodes = np.array(encodes)

        return encodes


class TokenizerMoirainBase(Tokenizer):
    def __init__(self, tokenizer_file):
            
            self.fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            num_added = self.fast_tokenizer.add_tokens(['<eos>', '<sos>', '<pad>'])
            
            self._toktypes = sorted(self.fast_tokenizer.vocab, key=self.fast_tokenizer.vocab.get)

            self._toktypes_order = {toktype: i for i, toktype in enumerate(self._toktypes)}
            self._toktypes_num = len(self._toktypes)

    
    @property
    def pad_idx(self):
        pad_idx = self.fast_tokenizer.convert_tokens_to_ids("<pad>")

        if pad_idx is None:
            raise ValueError(f"<pad> is not in the vocabulary")
        
        return pad_idx
    
    @property
    def sos_idx(self):
        sos_idx = self.fast_tokenizer.convert_tokens_to_ids("<sos>")

        if sos_idx is None:
            raise ValueError(f"<sos> is not in the vocabulary")
        
        return sos_idx
    
    @property
    def eos_idx(self):
        eos_idx = self.fast_tokenizer.convert_tokens_to_ids("<eos>")

        if eos_idx is None:
            raise ValueError(f"<eos> is not in the vocabulary")
        
        return eos_idx

    
    def encode(self, seq):
        return self.fast_tokenizer(seq).input_ids
    
    def decode(self, token_ids):
        return self.fast_tokenizer.convert_ids_to_tokens(token_ids)
    

class TokenizerMoirainMulti():
    def __init__(self, tokenizer_file):

        self.aa = Tokenizer(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', '<pad>'])

        self.na = TokenizerMoirainBase(tokenizer_file)