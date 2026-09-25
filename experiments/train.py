import os
import torch
import psutil
from functools import wraps

import logging

from datetime import datetime
from omegaconf import OmegaConf

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from experiments import utils as eu
from models import scheduler
from experiments import metrics


class ExperimentTrain:

    def __init__(self, *, conf):

        self.log = logging.getLogger(__name__)

        OmegaConf.set_struct(conf, False)

        # Configs
        self.conf = conf
        self.exp_conf = conf.experiment
        self.model_conf = conf.model
        self.data_conf = conf.data

        self.use_ddp = self.exp_conf.use_ddp

        if self.use_ddp :
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend='nccl', device_id=torch.cuda.current_device())
            self.ddp_info = eu.get_ddp_info()
            self.rank_log(f"GPU {self.ddp_info['local_rank']} is connected", all=True)
            self.num_replicas = self.ddp_info['world_size']
            self.rank = self.ddp_info['rank']
        else:
            self.num_replicas = None
            self.rank = None


        self.trained_steps = 0
        self.trained_epochs = 0
        
        # Initialize experiment objects
        if self.exp_conf.use_ckpt_conf:
            self.load_conf()

        self.tokenizer = self.build_tokenizer()
        self.models = self.build_models()

        if self.exp_conf.warm_start:
            self.warm_start()

        if self.model_conf.use_peft:
            self.models = self.peft_models()

        if self.exp_conf.freeze: 
            self.freeze()
        
        self.models = self.wrap_models()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        if self.exp_conf.resume_from_ckpt:
            self.load_ckpt()

        num_parameters = sum(p.numel() for model in self.models.values() for p in model.parameters())
        num_trainable_parameters = sum(p.numel() for model in self.models.values() for p in model.parameters() if p.requires_grad)
        self.exp_conf.num_parameters = num_parameters
        self.rank_log(f'Number of model parameters {num_parameters} ({num_trainable_parameters} trainable)')

        self.setup_checkpoint()

    
    def rank_log(self, msg, all=False):
        if not self.use_ddp:
            self.log.info(msg)
        else:
            if all:
                self.log.info(f"From rank { self.ddp_info['rank']}: {msg}")
            else:
                if self.ddp_info['rank'] in [0,-1]:
                    self.log.info(msg)


    @property
    def model(self):
        return self.models['model']

    @model.setter
    def model(self, value):
        self.models['model'] = value
        

    def build_tokenizer(self):
        raise NotImplementedError("Subclasses must implement build_tokenizer")
    

    def build_models(self):
        raise NotImplementedError("Subclasses must implement build_models")
    
    def peft_models(self):
        raise NotImplementedError("Subclasses must implement peft_models")
    
    def load_conf(self):
        raise NotImplementedError("Subclasses must implement load_conf")
    
    
    def wrap_models(self):
        wrapped = {}
        if not self.use_ddp:
            device = "cuda:0" if torch.cuda.is_available() and self.exp_conf.use_gpu else "cpu"
            self.device = torch.device(device)
            self.rank_log(f"Using device: {self.device}")
            for name, model in self.models.items():
                wrapped[name] = model.to(self.device)
        else:
            self.device = torch.device("cuda", self.ddp_info["local_rank"])
            self.rank_log(f"DDP mode on GPU {self.ddp_info['local_rank']} (node {self.ddp_info['node_id']})")
            for name, model in self.models.items():
                model = model.to(self.device)
                if any(p.requires_grad for p in model.parameters()):
                    model = DDP(model, device_ids=[self.ddp_info["local_rank"]], output_device=self.ddp_info["local_rank"])
                else:
                    for param in model.parameters():
                        dist.broadcast(param.data, src=0)
                    for buffer in model.buffers():
                        dist.broadcast(buffer.data, src=0)
                wrapped[name] = model
        return wrapped


    def warm_start(self):
        raise NotImplementedError("Subclasses must implement warm_start")

    
    def freeze(self):
        raise NotImplementedError("Subclasses must implement freeze")
    
        
    def build_optimizer(self):
        params = []
        for model in self.models.values():
            params += [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.exp_conf.learning_rate, betas=(0.9, 0.999), eps=1e-06, amsgrad=True)
    

    def build_scheduler(self):
        return scheduler.Scheduler(self.optimizer, self.exp_conf)
    
    @property
    def data_loader(self):
        raise NotImplementedError("Subclasses must implement data_loader")
    

    def set_mode(self, train=True):
        for model in self.models.values():
            model.train() if train else model.eval()
    

    def load_ckpt(self):
        ckpt_path = self.exp_conf.resume_from_ckpt
        self.rank_log(f'Loading ckpt from: {ckpt_path}')

        ckpt_pkl = eu.read_pkl(ckpt_path, use_torch=True)

        for name, model in self.models.items():
            eu.load_weights(model, ckpt_pkl, key=name)
        
        if 'optimizer' in ckpt_pkl:
            self.optimizer.load_state_dict(ckpt_pkl['optimizer'])
        if 'scheduler' in ckpt_pkl:
            self.scheduler.load_state_dict(ckpt_pkl['scheduler'])
        if 'epoch' in ckpt_pkl:
            self.trained_epochs = ckpt_pkl['epoch']
        if 'step' in ckpt_pkl:
            self.trained_steps = ckpt_pkl['step']

        del ckpt_pkl
        torch.cuda.empty_cache()

        
    def setup_checkpoint(self):
        dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        if self.exp_conf.ckpt_dir is not None:
            # Set-up checkpoint location
            ckpt_dir = os.path.join(
                self.exp_conf.ckpt_dir,
                self.exp_conf.name,
                dt_string)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            self.exp_conf.ckpt_dir = ckpt_dir
            self.rank_log(f'Checkpoints saved to: {ckpt_dir}')
        else:  
            self.rank_log('Checkpoint not being saved.')


    def take_ckpt(self):

        if self.rank == 0 or self.rank == None:

            ckpt_path = os.path.join(self.exp_conf.ckpt_dir, f'step_{self.trained_steps}.pth')

            model_dict = {name: model.module.state_dict() if isinstance(model, DDP) else model.state_dict() for name, model in self.models.items()}

            eu.write_checkpoint(
                ckpt_path,
                model_dict,
                self.conf,
                self.optimizer.state_dict(),
                self.scheduler.state_dict(),
                self.trained_epochs,
                self.trained_steps,
                use_torch=True
            )

            self.rank_log(f'Serialized experiment state to {ckpt_path}')

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
  

    def create_dataloaders(self):


        # Datasets
        train_dataset = self.data_loader.Dataset(self.data_conf, self.tokenizer, mode='train')
        valid_dataset = self.data_loader.Dataset(self.data_conf, self.tokenizer, mode='validation') if self.exp_conf.validate else None

        # Samplers
        train_sampler = self.data_loader.Sampler(self.data_conf, train_dataset, self.num_replicas, self.rank)
        valid_sampler = self.data_loader.Sampler(self.data_conf, valid_dataset, self.num_replicas, self.rank) if self.exp_conf.validate else None

        batch_size = self.exp_conf.batch_size // self.exp_conf.accumulation_steps

        if self.use_ddp:
            batch_size //= self.num_replicas

        # Loaders
        train_loader = self.data_loader.DataLoader(
            train_dataset,
            self.tokenizer,
            sampler=train_sampler,
            batch_size=batch_size,
            num_workers=self.exp_conf.num_loader_workers,
        )
        valid_loader = self.data_loader.DataLoader(
            valid_dataset,
            self.tokenizer,
            sampler=valid_sampler,
            batch_size=batch_size,
            num_workers=self.exp_conf.num_loader_workers,
        ) if self.exp_conf.validate else None

        return train_loader, valid_loader, train_sampler, valid_sampler



    def start_training(self):

        self.optimizer.zero_grad()

        train_loader, valid_loader, train_sampler, valid_sampler = self.create_dataloaders()

        self.rank_log(f'Training on {len(train_sampler)} samples')
        if valid_sampler is not None: self.rank_log(f'Validating on {len(valid_sampler)} samples')

        if self.exp_conf.resume_from_ckpt:
            train_sampler.set_start(self.trained_steps*(self.exp_conf.batch_size if not self.use_ddp else self.exp_conf.batch_size // self.num_replicas))

        start_epoch = self.trained_epochs

        for epoch in range(start_epoch, self.exp_conf.num_epoch):

            train_sampler.set_epoch(epoch)
            if valid_sampler is not None: valid_sampler.set_epoch(epoch)

            self.train_epoch(train_loader)

            self.trained_epochs += 1

            self.rank_log(f'End of training for epoch {self.trained_epochs}!')

            if valid_loader is not None: self.validate_epoch(valid_loader)

            if self.exp_conf.ckpt_dir is not None:
                self.take_ckpt()

        self.rank_log('Done')


    def get_batch_samples(self, data_iterator, num_batches):

        batch_samples = [next(data_iterator) for _ in range(num_batches)]
        num_items_in_batch = self.get_num_items_in_batch(batch_samples)

        return batch_samples, num_items_in_batch
    

    def get_num_items_in_batch(self, batch_samples):
        raise NotImplementedError("Subclasses must implement get_num_items_in_batch")
        

    def train_epoch(self, train_loader):

        self.set_mode(train=True)

        self.optimizer.zero_grad(set_to_none=True)

        tracker = metrics.MetricTracker(self.use_ddp)

        train_iterator = iter(train_loader)

        num_steps, remainder = divmod(len(train_loader), self.exp_conf.accumulation_steps)
        if remainder: 
            num_steps += 1

        for step in range(num_steps):

            num_microbatches = self.exp_conf.accumulation_steps
            if step == num_steps - 1 and remainder:
                num_microbatches = remainder

            batch_samples, num_items_in_batch = self.get_batch_samples(train_iterator, num_microbatches)

            if self.use_ddp:
                total_items = torch.as_tensor(num_items_in_batch, device=self.device, dtype=torch.float32).clone()
                dist.all_reduce(total_items, op=dist.ReduceOp.SUM)
            else:
                total_items = num_items_in_batch

            for train_feats, sample_ids in batch_samples:

                train_feats = {key: value.to(self.device) for key, value in train_feats.items()}

                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.exp_conf.use_amp):
                    loss, aux_data = self.loss_fn(train_feats)

                if torch.isnan(loss): raise Exception(f'NaN encountered')
                if torch.isinf(loss): raise Exception(f'Inf encountered')

                if self.use_ddp:
                    loss = loss * self.num_replicas / total_items
                else:
                    loss = loss / total_items

                loss.backward()

                tracker.update(aux_data)

            if self.exp_conf.get('max_gradient_norm', None) is not None:
                grad_norm = 0.0
                for model in self.models.values():
                    grad_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), self.exp_conf.max_gradient_norm)
                tracker.update({'grad_norm': (grad_norm.detach(), torch.tensor(1.0).to(self.device))})
                    
            self.optimizer.step()
    
            self.scheduler.step()
    
            self.optimizer.zero_grad(set_to_none=True)

            self.trained_steps += 1

            tracker.step()
        
            # Logging to terminal train loss
            if self.trained_steps == 1 or self.trained_steps % self.exp_conf.log_freq == 0:

                loss_log = tracker.get_log()

                self.rank_log(f'[Train {self.trained_steps}]: {loss_log}')

                tracker.reset()

            if self.exp_conf.ckpt_freq is not None and self.trained_steps % self.exp_conf.ckpt_freq == 0:
                if self.exp_conf.ckpt_dir is not None:
                    self.take_ckpt()


    def validate_epoch(self, valid_loader):

        self.set_mode(train=False)

        tracker = metrics.MetricTracker(self.use_ddp)

        valid_iterator = iter(valid_loader)

        num_steps, remainder = divmod(len(valid_loader), self.exp_conf.accumulation_steps)
        if remainder: 
            num_steps += 1

        for step in range(num_steps):

            num_microbatches = self.exp_conf.accumulation_steps
            if step == num_steps - 1 and remainder:
                num_microbatches = remainder

            batch_samples, num_items_in_batch = self.get_batch_samples(valid_iterator, num_microbatches)

            for valid_feats, sample_ids in batch_samples:

                valid_feats = {key: value.to(self.device) for key, value in valid_feats.items()}
                
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.exp_conf.use_amp):
                    loss, aux_data = self.loss_val_fn(valid_feats)

                tracker.update(aux_data)

            tracker.step()       

        # Logging to terminal validation loss
        loss_log = tracker.get_log()

        self.rank_log(f'[Validation {self.trained_epochs}]: {loss_log}')

        tracker.reset()

        
    

    @staticmethod
    def detach_outputs(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result, info = func(*args, **kwargs)
            for k, (v, s) in info.items():
                info[k] = (v.detach(), s.detach() if s is not None else None)
            return result, info
        return wrapper


    def loss_fn(self, batch):
        raise NotImplementedError("Subclasses must implement loss_fn")
    
    def loss_val_fn(self, batch):
        return self.loss_fn(batch)