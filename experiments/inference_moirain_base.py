import os
import numpy as np
import hydra
import torch
from omegaconf import DictConfig

from experiments import inference

from models import model_moirain_base
from data import dataloader_moirain_base
from data import tokenizers



class ExperimentMoirainBaseInference(inference.ExperimentInference):


    def build_tokenizer(self):
        return tokenizers.TokenizerMoirainBase(self.data_conf.tokenizer_path)

    def build_model(self):
        return model_moirain_base.MainModel(self.model_conf, self.tokenizer)
        
    @property
    def data_loader(self):
        return dataloader_moirain_base
    


    def inference_fn(self, test_loader):


        for test_feats, sample_id in test_loader:

            test_feats = {key: value.to(self.device) for key, value in test_feats.items()}

            sampled_output = self.sample(test_feats)

            sampled_seq = sampled_output['seq_na']
            sampled_prob = np.mean(sampled_output['probs_na'][:-1])

            seq_na_str_uncut = '|'.join(sampled_seq)
            seq_na_str = ''.join([val for val in sampled_seq if val != '<sos>' and val != '<eos>'])

            with open(os.path.join(self.output_dir, 'seq_na_uncut.fasta'), 'a') as f:
                f.write(f'>{sample_id[0]},{sampled_prob:.4f}\n')
                f.write(f'{seq_na_str_uncut}\n')

            with open(os.path.join(self.output_dir, 'seq_na.fasta'), 'a') as f:
                f.write(f'>{sample_id[0]},{sampled_prob:.4f}\n')
                f.write(f'{seq_na_str}\n')
            
            if int(sample_id[0]) % 100 == 0:
                self.rank_log(f'Done sample {sample_id[0]}', all=True)
            
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
    
    

    def sample(self, batch):
        self.len = 0
        self.eos_id = torch.tensor(self.tokenizer.encode("<eos>")).to(batch['ttype_na'])

        probs_na = []

        while not (batch['ttype_na'][0][-1] == self.eos_id):

            model_out = self.model(batch)

            tokens_ids, tokens_probs = self.sample_tokens(model_out['logits_type_na'][..., -1:, :])

            batch = self.update_batch(batch, tokens_ids)

            probs_na.append(tokens_probs[0][-1].tolist())

        seq_na = self.tokenizer.decode(batch['ttype_na'][0])

        res = {
            'seq_na': seq_na,
            'probs_na': probs_na
        }

        return res

 

@hydra.main(version_base=None, config_path="../configs", config_name="inference_moirain_base")
def run(conf: DictConfig) -> None:

    with torch.inference_mode():
        inference = ExperimentMoirainBaseInference(conf)
        inference.run_inference()

if __name__ == '__main__':
    run()