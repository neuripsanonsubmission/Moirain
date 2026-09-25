import torch
import torch.nn.functional as F
import hydra
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP

from experiments import train

from data import tokenizers
from data import dataloader_moirain
from experiments import utils as eu

from models import model_moirain


def selective_log_softmax(logits, targets):

    log_probs = F.log_softmax(logits, dim=-1)

    log_probs_target = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    return log_probs_target


class ExperimentMoirainTrain(train.ExperimentTrain):

    def build_tokenizer(self):
        return tokenizers.TokenizerMoirainMulti(self.data_conf.tokenizer_path)

    def build_models(self):
        return {'model': model_moirain.MainModel(self.model_conf, self.tokenizer)}
    
    @property
    def data_loader(self):
        return dataloader_moirain
    
    def load_conf(self):

        ckpt_pkl = eu.read_pkl(self.exp_conf.warm_start, use_torch=True)
        model_conf = ckpt_pkl['conf']['model']
        
        self.model_conf.na = model_conf.na
        self.model_conf.aa = model_conf.aa
        self.model_conf.cross = model_conf.cross

    def warm_start(self):
        
        ckpt_pkl = eu.read_pkl(self.exp_conf.warm_start, use_torch=True)
        
        eu.load_weights(self.model, ckpt_pkl, strict=True)

        self.rank_log(f'Warm starting from: {self.exp_conf.warm_start}')


    def peft_models(self):
        config = LoraConfig(
            r=self.model_conf.peft_rank,
            lora_alpha=self.model_conf.peft_alpha,
            target_modules=["linear_q", "linear_kv", "linear_out", "feed_forward.0", "feed_forward.2"],
            lora_dropout=0.0,
            bias="none"
        )
        
        model = get_peft_model(self.model, config)
            
        return {'model': model}
    

    def get_logps(self, batch):

        model_out = self.model(batch)

        batch_size, num_tok, num_c = model_out['logits_type_na'].shape

        pad_mask = batch['pad_na'][:,1:]

        logits = model_out['logits_type_na'][:,:-1]

        target = batch['ttype_na'][:,1:]*pad_mask

        target_logps = (selective_log_softmax(logits, target)*pad_mask).sum(-1)

        return target_logps[:batch_size//2], target_logps[batch_size//2:]
    

    def get_ref_logps(self, batch):
        ctx = self.model.module.disable_adapter() if isinstance(self.model, DDP) else self.model.disable_adapter()
        with torch.no_grad() and ctx:
            ref_chosen_logps, ref_rejected_logps = self.get_logps(batch)

        return ref_chosen_logps, ref_rejected_logps
    

    def dpo_loss(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps):

        loss = -F.logsigmoid(self.exp_conf.beta * (chosen_logps - rejected_logps - ref_chosen_logps + ref_rejected_logps))

        chosen_rewards = (chosen_logps - ref_chosen_logps)
        rejected_rewards = (rejected_logps - ref_rejected_logps)

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        batch_size = chosen_logps.shape[0]

        aux_data = {
            "total_loss": (loss.sum(), torch.tensor(batch_size).to(self.device)),
            "rewards_chosen": (chosen_rewards.sum(), torch.tensor(batch_size).to(self.device)),
            "rewards_rejected": (rejected_rewards.sum(), torch.tensor(batch_size).to(self.device)),
            "rewards_accuracies": (reward_accuracies.sum(), torch.tensor(batch_size).to(self.device)),
        }

        return loss, aux_data
    

    def ipo_loss(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps):

        delta = chosen_logps - rejected_logps - ref_chosen_logps + ref_rejected_logps - 0.5 / self.exp_conf.beta
        loss = delta * delta

        chosen_rewards = (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = (rejected_logps - ref_rejected_logps).detach()

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        batch_size = chosen_logps.shape[0]

        aux_data = {
            "total_loss": (loss.sum(), torch.tensor(batch_size).to(self.device)),
            "rewards_chosen": (chosen_rewards.sum(), torch.tensor(batch_size).to(self.device)),
            "rewards_rejected": (rejected_rewards.sum(), torch.tensor(batch_size).to(self.device)),
            "rewards_accuracies": (reward_accuracies.sum(), torch.tensor(batch_size).to(self.device)),
        }

        return loss, aux_data


    def suan_loss(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps):
    
        sup_loss = - F.logsigmoid(chosen_logps - rejected_logps + self.exp_conf.tau)

        unsup_loss = (F.sigmoid(chosen_logps - ref_chosen_logps) - 0.5).detach()*chosen_logps + (F.sigmoid(rejected_logps - ref_rejected_logps) - 0.5).detach()*rejected_logps

        loss = sup_loss + self.exp_conf.beta * unsup_loss

        chosen_rewards = (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = (rejected_logps - ref_rejected_logps).detach()

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        batch_size = chosen_logps.shape[0]

        loss_to_log = - F.logsigmoid(chosen_logps - rejected_logps + self.exp_conf.tau) + self.exp_conf.beta * torch.log(torch.cosh(torch.log(torch.tensor(0.5)) + chosen_logps - ref_chosen_logps))# + self.exp_conf.beta * torch.log(torch.cosh(torch.log(torch.tensor(0.5)) + rejected_logps - ref_rejected_logps))

        aux_data = {
            "total_loss": (loss_to_log.sum(), torch.tensor(batch_size).to(self.device)),
            "rewards_chosen": (chosen_rewards.sum(), torch.tensor(batch_size).to(self.device)),
            "rewards_rejected": (rejected_rewards.sum(), torch.tensor(batch_size).to(self.device)),
            "rewards_accuracies": (reward_accuracies.sum(), torch.tensor(batch_size).to(self.device)),
        }

        return loss, aux_data


    def get_num_items_in_batch(self, batch_samples):
            return sum(batch['pad_na'].shape[0]//2 for batch, sample_ids in batch_samples)


    @train.ExperimentTrain.detach_outputs
    def loss_fn(self, batch):

        chosen_logps, rejected_logps = self.get_logps(batch)
        ref_chosen_logps, ref_rejected_logps = self.get_ref_logps(batch)

        if self.exp_conf.loss_type == 'dpo':
            final_loss, aux_data = self.dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps)
        elif self.exp_conf.loss_type == 'ipo':
            final_loss, aux_data = self.ipo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps)
        elif self.exp_conf.loss_type == 'suan':
            final_loss, aux_data = self.suan_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps)
        else:
            raise ValueError(f'Invalid loss type: {self.exp_conf.loss_type}')
        
        return final_loss.sum(), aux_data
    


@hydra.main(version_base=None, config_path="../configs", config_name="base_moirain")
def run(conf):
    exp = ExperimentMoirainTrain(conf=conf)
    exp.start_training()
    


if __name__ == '__main__':
    run()