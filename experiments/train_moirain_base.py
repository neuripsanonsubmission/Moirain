import torch
import torch.nn.functional as F
import hydra
from peft import LoraConfig, get_peft_model

from experiments import train

from data import tokenizers
from data import dataloader_moirain_base
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


    def build_optimizer(self):

        decay = []
        no_decay = []

        for model in self.models.values():
            seen = set()  # per model — only deduplicates within one model
            for name, param in model.named_parameters():
                if not param.requires_grad or id(param) in seen:
                    continue
                seen.add(id(param))
                if param.ndim <= 1 or name.endswith('.bias'):
                    no_decay.append(param)
                else:
                    decay.append(param)

        params = [{"params": decay, "weight_decay": self.exp_conf.get('weight_decay', 0.1)}, {"params": no_decay, "weight_decay": 0.0}]

        return torch.optim.AdamW(params, lr=self.exp_conf.learning_rate, betas=(0.9, 0.95), eps=1e-06)


    def get_num_items_in_batch(self, batch_samples):
        return sum(batch['pad_na'][:, 1:].sum() for batch, sample_ids in batch_samples)


    @train.ExperimentTrain.detach_outputs
    def loss_fn(self, batch):
        
        model_out = self.model(batch)

        batch_size, num_tok, num_c = model_out['logits_type_na'].shape

        pad_mask = batch['pad_na'][:,1:]

        mask = pad_mask

        logits_type = model_out['logits_type_na'][:,:-1].reshape(-1, num_c)

        target_type = (batch['ttype_na'][:,1:]*pad_mask).reshape(-1)
        
        loss_type = F.cross_entropy(logits_type, target_type, reduction="none").reshape(batch_size, num_tok-1)

        final_loss_type = (loss_type*mask).sum()
        
        aux_data = {
            'total_loss': (final_loss_type, mask.sum())
        }
        
        return final_loss_type, aux_data
    


@hydra.main(version_base=None, config_path="../configs", config_name="base_moirain_base")
def run(conf):
    exp = ExperimentRLMTrain(conf=conf)
    exp.start_training()
    


if __name__ == '__main__':
    run()