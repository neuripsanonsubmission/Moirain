import torch
import numpy as np
import collections

from data import dataloader

class Dataset(dataloader.Dataset):

    def produce_sample_train(self, seq):
        
        ttype_na = ['U' if val=='T' else val for val in seq]
        ttype_na = "<sos>" + "".join(ttype_na) + "<eos>"
        ttype_na = self.tokenizer.encode(ttype_na)
        tpos_na = np.arange(0, len(ttype_na))

        if len(ttype_na) > self.max_len+2:

            crop_center = np.random.choice(range(self.max_len//2, len(ttype_na)-self.max_len//2))
            tpos_na = tpos_na[crop_center-self.max_len//2:crop_center+self.max_len//2]
            ttype_na = ttype_na[crop_center-self.max_len//2:crop_center+self.max_len//2]


        final_feats = {
            'ttype_na': torch.tensor(ttype_na).to(torch.int),
            'tpos_na': torch.tensor(tpos_na).to(torch.int16)
        }

        return final_feats
    
    def produce_sample_test(self):

        ttype_na = "<sos>"
        tpos_na = [0] 

        ttype_na = self.tokenizer.encode(ttype_na)

        final_feats = {
            'ttype_na': torch.tensor(ttype_na).to(torch.int),
            'tpos_na': torch.tensor(tpos_na).to(torch.int16)
        }
        
        return final_feats
        



class Sampler(dataloader.Sampler):
    pass

class TestSampler(dataloader.TestSampler):
    pass


class DataLoader(dataloader.DataLoader):

    def cat_features(self, batch):

        pad_na_ind = self.tokenizer.pad_idx
        
        combined_dict = collections.defaultdict(list)
        names=[]
        lengths_na=[]

        for chain_dict in batch:
            for feat_name, feat_val in chain_dict[0].items():
                feat_val = feat_val[None]
                combined_dict[feat_name].append(feat_val)
            
            names.append(chain_dict[1])
            lengths_na.append(chain_dict[0]['ttype_na'].shape[-1])

        pad_length_na = max(lengths_na)

        for feat_name, feat_vals in combined_dict.items():

            pad_feat_vals = self.pad_features(feat_name, feat_vals, pad_length_na, pad_na_ind)
            
            if isinstance(feat_vals[0], torch.Tensor):
                combined_dict[feat_name] = torch.cat(pad_feat_vals, dim=0)
            else:
                raise ValueError(f'Invalid feature instance: {type(feat_vals[0])}')

        pad_na = []
        for n in lengths_na:
            pad_na.append(torch.cat((torch.ones(n), torch.zeros(pad_length_na - n)))[None])
        combined_dict['pad_na'] = torch.cat(pad_na, dim=0).to(torch.int)

        return (combined_dict, names)



    def pad_features(self, name, vals, pad_length, pad_ind):

        padded_vals = []
        
        if isinstance(vals[0], torch.Tensor):
            if 'pos' in name:
                for val in vals:
                    fill_size = pad_length-val.shape[-1]
                    fill_tensor = torch.full((1, fill_size), torch.tensor(0, requires_grad=False).to(val))
                    padded_vals.append(torch.cat((val, fill_tensor), dim=-1))
            elif 'type' in name:
                for val in vals:
                    fill_size = pad_length-val.shape[-1]
                    fill_tensor = torch.full((1, fill_size), torch.tensor(pad_ind, requires_grad=False).to(val))
                    padded_vals.append(torch.cat((val, fill_tensor), dim=-1))
            elif 'mask' in name:
                for val in vals:
                    fill_size = pad_length-val.shape[-1]
                    fill_tensor = torch.full((1, fill_size), torch.tensor(0, requires_grad=False).to(val))
                    padded_vals.append(torch.cat((val, fill_tensor), dim=-1))
            else:
                raise ValueError(f'Invalid feature name: {name}')
            
        else:
            raise ValueError(f'Invalid feature instance: {type(vals[0])}')

        return padded_vals 