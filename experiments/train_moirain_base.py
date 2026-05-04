import torch
import torch.nn.functional as F
import hydra
from peft import LoraConfig, get_peft_model

from experiments import train

from data import tokenizers
from data import dataloader_moirain_base
from data import utils as du
from experiments import utils as eu

from models import model_moirain_base



class ExperimentMoirainBaseTrain(train.ExperimentTrain):

    def build_tokenizer(self):
        return tokenizers.TokenizerMoirainBase(self.data_conf.tokenizer_path)

    def build_models(self):
        return {'model': model_moirain_base.MainModel(self.model_conf, self.tokenizer)}
    
    def peft_models(self):
        config = LoraConfig(
            r=self.model_conf.peft_rank,
            lora_alpha=self.model_conf.peft_alpha,
            target_modules=["linear_q", "linear_kv", "linear_out", "feed_forward.0", "feed_forward.2"],
            lora_dropout=0.0,
            bias="none"
        )
        return {'model': get_peft_model(self.model, config)}

    @property
    def data_loader(self):
        return dataloader_moirain_base

    def warm_start(self):
        
        ckpt_pkl = eu.read_pkl(self.exp_conf.warm_start, use_torch=True)
        
        eu.load_weights(self.model, ckpt_pkl, strict=False)

        self.rank_log(f'Warm starting from: {self.exp_conf.warm_start}')


    @train.ExperimentTrain.detach_outputs
    def loss_fn(self, batch):
        
        model_out = self.model(batch)

        batch_size, num_tok, num_c = model_out['logits_type_na'].shape

        pad_mask = batch['pad_na'][:,1:]

        mask = pad_mask

        logits_type = model_out['logits_type_na'][:,:-1].reshape(-1, num_c)

        target_type = (batch['ttype_na'][:,1:]*pad_mask).reshape(-1)
        
        loss_type = F.cross_entropy(logits_type, target_type, reduction="none").reshape(batch_size, num_tok-1)

        final_loss_type = (loss_type*mask).sum(-1) / mask.sum(-1)

        final_loss = final_loss_type
        
        aux_data = {
            'total_loss': (final_loss.sum(), torch.tensor(batch_size).to(self.device)),
            'type_loss': (final_loss_type.sum(), torch.tensor(batch_size).to(self.device))
        }
        
        return final_loss.sum()/batch_size, aux_data
    


@hydra.main(version_base=None, config_path="../configs", config_name="base_moirain_base")
def run(conf):
    exp = ExperimentMoirainBaseTrain(conf=conf)
    exp.start_training()
    


if __name__ == '__main__':
    run()
        
    


