import torch
import numpy as np
import collections
import gemmi

from data import dataloader
from data import rigid_utils as ru
from data import utils as du


class Dataset(dataloader.Dataset):

    def produce_sample_train(self, csv_row):

        structure = du.read_structure_from_sql(csv_row.id_aa, f'{self.data_path}/dataset_aa.db')
        sequence = du.read_sequence_from_sql(csv_row.id_na, f'{self.data_path}/dataset_na.db')
        
        ttype_na = ['U' if val=='T' else val for val in sequence]
        ttype_na = "<sos>" + "".join(ttype_na) + "<eos>"
        ttype_na = self.tokenizer.na.encode(ttype_na)
        tpos_na = np.arange(0, len(ttype_na))
        
        feats_aa = gemmi.make_structure_from_block(gemmi.cif.read_string(structure).sole_block())[0]["A"]

        tpos_aa = np.array([r.seqid.num for r in feats_aa])
        tpos_aa = tpos_aa - np.min(tpos_aa)

        ttype_aa = [r.name for r in feats_aa]
        ttype_aa = self.tokenizer.aa.encode(ttype_aa)

        n_atoms, ca_atoms, c_atoms, plddt = map(torch.tensor, zip(*[(r.sole_atom("N").pos.tolist(), r.sole_atom("CA").pos.tolist(), r.sole_atom("C").pos.tolist(), r.sole_atom("CA").b_iso) for r in feats_aa]))
        com = ca_atoms.mean(0)
        n_atoms -= com
        ca_atoms -= com
        c_atoms -= com

        T_aa = ru.Rigid.from_3_points(n_atoms, ca_atoms, c_atoms)
        T_aa = T_aa.apply_trans_fn(lambda x: x * 0.1)

        final_feats = {
            'ttype_na': torch.tensor(ttype_na).to(torch.int),
            'tpos_na': torch.tensor(tpos_na).to(torch.int16),
            'ttype_aa': torch.tensor(ttype_aa).to(torch.int),
            'tpos_aa': torch.tensor(tpos_aa).to(torch.int16),
            'plddt_aa': plddt,
            'T_aa': T_aa
        }

        return final_feats
    

    def produce_sample_test(self, csv_row):

        structure = du.read_structure_from_sql(csv_row.id_aa, self.data_path)
        
        ttype_na = "<sos>"
        tpos_na = [0] 

        ttype_na = self.tokenizer.na.encode(ttype_na)
        
        feats_aa = gemmi.make_structure_from_block(gemmi.cif.read_string(structure).sole_block())[0]["A"]

        tpos_aa = np.array([r.seqid.num for r in feats_aa])
        tpos_aa = tpos_aa - np.min(tpos_aa)

        ttype_aa = [r.name for r in feats_aa]
        ttype_aa = self.tokenizer.aa.encode(ttype_aa)

        n_atoms, ca_atoms, c_atoms, plddt = map(torch.tensor, zip(*[(r.sole_atom("N").pos.tolist(), r.sole_atom("CA").pos.tolist(), r.sole_atom("C").pos.tolist(), r.sole_atom("CA").b_iso) for r in feats_aa]))
        com = ca_atoms.mean(0)
        n_atoms -= com
        ca_atoms -= com
        c_atoms -= com

        T_aa = ru.Rigid.from_3_points(n_atoms, ca_atoms, c_atoms)
        T_aa = T_aa.apply_trans_fn(lambda x: x * 0.1)


        final_feats = {
            'ttype_na': torch.tensor(ttype_na).to(torch.int),
            'tpos_na': torch.tensor(tpos_na).to(torch.int16),
            'ttype_aa': torch.tensor(ttype_aa).to(torch.int),
            'tpos_aa': torch.tensor(tpos_aa).to(torch.int16),
            'plddt_aa': plddt,
            'T_aa': T_aa
        }

        return final_feats
        



class Sampler(dataloader.Sampler):
    pass

class TestSampler(dataloader.TestSampler):
    pass


class DataLoader(dataloader.DataLoader):

    def cat_features(self, batch):

        pad_na_ind = self.tokenizer.na.pad_idx
        pad_aa_ind = self.tokenizer.aa.pad_idx
        
        combined_dict = collections.defaultdict(list)
        names=[]
        lengths_na=[]
        lengths_aa=[]

        for chain_dict in batch:
            for feat_name, feat_val in chain_dict[0].items():
                feat_val = feat_val[None]
                combined_dict[feat_name].append(feat_val)
            
            names.append(chain_dict[1])
            lengths_na.append(chain_dict[0]['ttype_na'].shape[-1])
            lengths_aa.append(chain_dict[0]['ttype_aa'].shape[-1])

        pad_length_na = max(lengths_na)
        pad_length_aa = max(lengths_aa)

        for feat_name, feat_vals in combined_dict.items():

            if 'na' in feat_name:
                pad_feat_vals = self.pad_features(feat_name, feat_vals, pad_length_na, pad_na_ind)
            elif 'aa' in feat_name:
                pad_feat_vals = self.pad_features(feat_name, feat_vals, pad_length_aa, pad_aa_ind)
            
            if isinstance(feat_vals[0], torch.Tensor):
                combined_dict[feat_name] = torch.cat(pad_feat_vals, dim=0)
            elif isinstance(feat_vals[0], ru.Rigid):
                combined_dict[feat_name] = ru.Rigid.cat(pad_feat_vals, dim=0)
            else:
                raise ValueError(f'Invalid feature instance: {type(feat_vals[0])}')

        pad_na = []
        for n in lengths_na:
            pad_na.append(torch.cat((torch.ones(n), torch.zeros(pad_length_na - n)))[None])
        combined_dict['pad_na'] = torch.cat(pad_na, dim=0).to(torch.int)

        pad_aa = []
        for n in lengths_aa:
            pad_aa.append(torch.cat((torch.ones(n), torch.zeros(pad_length_aa - n)))[None])
        combined_dict['pad_aa'] = torch.cat(pad_aa, dim=0).to(torch.int)

        return (combined_dict, names)



    def pad_features(self, name, vals, pad_length, pad_ind):

        padded_vals = []
        
        if isinstance(vals[0], torch.Tensor):
            if 'pos' in name:
                for val in vals:
                    fill_size = pad_length-val.shape[-1]
                    fill_tensor = torch.full((1, fill_size), torch.tensor(0, requires_grad=False).to(val))
                    padded_vals.append(torch.cat((val, fill_tensor), dim=-1))
            elif 'plddt' in name:
                for val in vals:
                    fill_size = pad_length-val.shape[-1]
                    fill_tensor = torch.full((1, fill_size), torch.tensor(0.0, requires_grad=False).to(val))
                    padded_vals.append(torch.cat((val, fill_tensor), dim=-1))
            elif 'type' in name:
                for val in vals:
                    fill_size = pad_length-val.shape[-1]
                    fill_tensor = torch.full((1, fill_size), torch.tensor(pad_ind, requires_grad=False).to(val))
                    padded_vals.append(torch.cat((val, fill_tensor), dim=-1))
            else:
                raise ValueError(f'Invalid feature name: {name}')

        elif isinstance(vals[0], ru.Rigid):
            for val in vals:
                fill_size = pad_length-val.shape[-1]
                fill_tensor = ru.Rigid.identity((1,fill_size), requires_grad=False)
                padded_vals.append(ru.Rigid.cat((val, fill_tensor), dim=-1))
        else:
            raise ValueError(f'Invalid feature instance: {type(vals[0])}')

        return padded_vals 