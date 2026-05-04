import torch
import torch.nn.functional as F
import hydra
from peft import LoraConfig, get_peft_model

from experiments import train

from data import tokenizers
from data import dataloader_moirain_multi
from experiments import utils as eu

from models import model_moirain_multi



class ExperimentMoirainMultiTrain(train.ExperimentTrain):

    def build_tokenizer(self):
        return tokenizers.TokenizerMoirainMulti(self.data_conf.tokenizer_path)

    def build_models(self):
        return {'model': model_moirain_multi.MainModel(self.model_conf, self.tokenizer)}
    
    @property
    def data_loader(self):
        return dataloader_moirain_multi
    
    def peft_models(self):

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

        for name, param in self.model.named_parameters():
            if "encoder_aa" in name or "encoder_cross" in name or "type_aa_embedder" in name:
                param.requires_grad = True
            
        return {'model': self.model}

    def warm_start(self):
        
        ckpt_pkl = eu.read_pkl(self.exp_conf.warm_start, use_torch=True)
        
        eu.load_weights(self.model, ckpt_pkl, strict=False)

        self.rank_log(f'Warm starting from: {self.exp_conf.warm_start}')

    def freeze(self):
        enc = self.model.type_embedder
        
        for p in enc.type_embedder_na.parameters():
            p.requires_grad = False
        for p in enc.layer_norm.parameters():
            p.requires_grad = False

        for b in range(self.model_conf.na.num_blocks_na):
            enc = self.model.trunk[f'encoder_na_{b}']
            
            for p in enc.self_mha.parameters():
                p.requires_grad = False
            for p in enc.feed_forward.parameters():
                p.requires_grad = False
            for p in enc.layer_norm_1.parameters():
                p.requires_grad = False
            for p in enc.layer_norm_2.parameters():
                p.requires_grad = False

        for p in self.model.log_head_type.parameters():
            p.requires_grad = False
    

    def tofu_loss(self, logits, targets):

        self.beta = 0.8
        self.gamma = 3
        
        log_probs_beta = F.log_softmax(logits / self.beta, dim=-1)
        probs = F.log_softmax(logits, dim=-1).exp()

        p = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        log_p_b = log_probs_beta.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        term = ((1-p)**self.gamma - self.gamma*p*(1-p)**(self.gamma-1)*torch.log(p)).detach()
        loss = -term * log_p_b * self.beta

        return loss


    @train.ExperimentTrain.detach_outputs
    def loss_fn(self, batch):
        
        model_out = self.model(batch)

        batch_size, num_tok, num_c = model_out['logits_type_na'].shape

        pad_mask = batch['pad_na'][:,1:]

        mask = pad_mask

        logits_type = model_out['logits_type_na'][:,:-1].reshape(-1, num_c)

        target_type = (batch['ttype_na'][:,1:]*pad_mask).reshape(-1)

        if self.exp_conf.loss_type == 'ce':
            loss_type = F.cross_entropy(logits_type, target_type, reduction="none").reshape(batch_size, num_tok-1)
        elif self.exp_conf.loss_type == 'tofu':
            loss_type = self.tofu_loss(logits_type, target_type).reshape(batch_size, num_tok-1)
        else:
            raise ValueError(f'Invalid loss type: {self.exp_conf.loss_type}')

        final_loss_type = (loss_type*mask).sum(-1) / mask.sum(-1)

        final_loss = final_loss_type
        
        aux_data = {
            'total_loss': (final_loss.sum(), torch.tensor(batch_size).to(self.device)),
            'type_loss': (final_loss_type.sum(), torch.tensor(batch_size).to(self.device))
        }
        
        return final_loss.sum()/batch_size, aux_data
    


@hydra.main(version_base=None, config_path="../configs", config_name="base_moirain_multi")
def run(conf):
    exp = ExperimentMoirainMultiTrain(conf=conf)
    exp.start_training()
    


if __name__ == '__main__':
    run()
        
    


