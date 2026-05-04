import os
import argparse
import numpy as np
import torch
import gemmi
import logging
from omegaconf import OmegaConf

from peft import LoraConfig, get_peft_model

from experiments import inference

from models import model_moirain_multi
from data import rigid_utils as ru
from data import tokenizers



class ExperimentMoirainMultiInference(inference.ExperimentInference):


    def build_tokenizer(self):
        return tokenizers.TokenizerMoirainMulti(self.data_conf.tokenizer_path)

    def build_model(self):
        return model_moirain_multi.MainModel(self.model_conf, self.tokenizer)

    def peft_model(self):

        def layers_to_peft(i):
            base = f"trunk.encoder_na_{i}"
            return (
                f"{base}.self_mha.linear_q",
                f"{base}.self_mha.linear_kv",
                f"{base}.self_mha.linear_out",
                f"{base}.feed_forward.0",
                f"{base}.feed_forward.2",
            )
        
        config = LoraConfig(
            r=self.model_conf.peft_rank,
            lora_alpha=self.model_conf.peft_alpha,
            target_modules=[s for i in range(self.model_conf.na.num_blocks_na) for s in layers_to_peft(i)],
            lora_dropout=0.0,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, config)
            
        return self.model
    


    def create_test_loader(self):
        return None
    

    def produce_sample_test(self):
        
        ttype_na = "<sos>"
        tpos_na = [0] 

        ttype_na = self.tokenizer.na.encode(ttype_na)
        
        feats_aa = gemmi.read_structure(self.data_conf.cif_path)[0]["A"]

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
            'ttype_na': torch.tensor(ttype_na).to(torch.int)[None],
            'tpos_na': torch.tensor(tpos_na).to(torch.int16)[None],
            'ttype_aa': torch.tensor(ttype_aa).to(torch.int)[None],
            'tpos_aa': torch.tensor(tpos_aa).to(torch.int16)[None],
            'plddt_aa': plddt[None],
            'T_aa': T_aa,
            'pad_aa': torch.ones(len(ttype_aa))[None],
            'pad_na': torch.ones(len(ttype_na))[None],
        }

        return final_feats
    


    def inference_fn(self, test_loader):


        test_feats = self.produce_sample_test()
        test_feats = {key: value.to(self.device) for key, value in test_feats.items()}

        s_aa = self.model.forward_aa(test_feats)

        for idx in range(self.inf_conf.nof_samples):

            copy_test_feats = {key: value.clone() for key, value in test_feats.items()}
            sampled_output = self.sample(copy_test_feats, s_aa)

            sampled_seq = sampled_output['seq_na']
            sampled_prob = np.mean(sampled_output['probs_na'][:-1])

            seq_na_str_uncut = '|'.join(sampled_seq)
            seq_na_str = ''.join([val for val in sampled_seq if val != '<sos>' and val != '<eos>'])

            with open(os.path.join(self.output_dir, 'seq_na_uncut.fasta'), 'a') as f:
                f.write(f'>sample_{idx},{sampled_prob:.4f}\n')
                f.write(f'{seq_na_str_uncut}\n')

            with open(os.path.join(self.output_dir, 'seq_na.fasta'), 'a') as f:
                f.write(f'>sample_{idx},{sampled_prob:.4f}\n')
                f.write(f'{seq_na_str}\n')

        self.rank_log(f'Done sample!', all=True)
            
        return 0


    
    def update_batch(self, batch, new_token_ids):

        if batch['ttype_na'][0][-1] != self.eos_id:

            self.len += 1

            batch['tpos_na'] = torch.cat((batch['tpos_na'], batch['tpos_na'][...,-1:]+1), dim=-1)

            if self.len <= self.max_len:
                batch['ttype_na'] = torch.cat((batch['ttype_na'], new_token_ids), dim=-1)
            else:
                eos_to_append = self.eos_id.repeat(batch['ttype_na'].shape[:-1]).to(batch['ttype_na'])
                batch['ttype_na'] = torch.cat((batch['ttype_na'], eos_to_append.unsqueeze(-1)), dim=-1)
                
            batch['pad_na'] = torch.cat((batch['pad_na'][...,:1], batch['pad_na']), dim=-1)

        return batch
    
    

    def sample(self, batch, s_aa):
        self.len = 0
        self.eos_id = torch.tensor(self.tokenizer.na.encode("<eos>")).to(batch['ttype_na'])

        probs_na = []

        while not (batch['ttype_na'][0][-1] == self.eos_id):

            model_out = self.model.forward_na(batch, s_aa)

            tokens_ids, tokens_probs = self.sample_tokens(model_out['logits_type_na'][..., -1:, :])

            batch = self.update_batch(batch, tokens_ids)

            probs_na.append(tokens_probs[0][-1].tolist())

        seq_na = self.tokenizer.na.decode(batch['ttype_na'][0])

        res = {
            'seq_na': seq_na,
            'probs_na': probs_na
        }

        return res

 

def parse_args():
    parser = argparse.ArgumentParser(description="Moirain Multi Inference")

    # inference
    parser.add_argument("--inference.name", type=str, default="moirain_dpo")
    parser.add_argument("--inference.seed", type=int, default=123)
    parser.add_argument("--inference.nof_samples", type=int, default=1000)
    parser.add_argument("--inference.use_ddp", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--inference.sample_mode", type=str,   default="topp")
    parser.add_argument("--inference.temperature", type=float, default=1.0)
    parser.add_argument("--inference.kp", type=float, default=0.9)
    parser.add_argument("--inference.max_len", type=int, default=50)
    parser.add_argument("--inference.num_loader_workers", type=int, default=1)
    parser.add_argument("--inference.output_dir", type=str, default="./inference_outputs/")
    parser.add_argument("--inference.model_path", type=str, default="./ckpt/moirain_multi/step_21174.pth")

    # data
    parser.add_argument("--data.cif_path", type=str, default=None)
    parser.add_argument("--data.tokenizer_path", type=str, default="./data/tokenizer.json")

    return parser.parse_args()


def build_conf_from_args(args):
    args_dict = vars(args)

    conf = OmegaConf.create({"inference": {}, "data": {}})

    for key, value in args_dict.items():
        section, field = key.split(".", 1)
        conf[section][field] = value

    return conf


def run():

    # Re-add logging setup that Hydra was handling automatically
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        handlers=[
            logging.StreamHandler(),  # logs to terminal
        ]
    )

    args = parse_args()
    conf = build_conf_from_args(args)

    with torch.inference_mode():
        experiment = ExperimentMoirainMultiInference(conf)
        experiment.run_inference()


if __name__ == '__main__':
    run()