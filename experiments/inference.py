import os
import torch
import torch.nn.functional as F
import logging
from datetime import datetime

from omegaconf import OmegaConf

import torch.distributed as dist

from experiments import utils as eu







class ExperimentInference:

    def __init__(
            self,
            conf
        ):

        self.log = logging.getLogger(__name__)

        # Remove static type checking.
        OmegaConf.set_struct(conf, False)

        # Prepare configs.
        self.conf = conf
        self.inf_conf = conf.inference
        self.data_conf = conf.data

        self.sample_mode = conf.inference.sample_mode
        self.max_len = conf.inference.max_len
        self.kp = conf.inference.kp
        self.temperature = conf.inference.temperature

        # Set-up directories
        self.weights_path = self.inf_conf.model_path

        if self.inf_conf.name is None:
            name_string = self.inf_conf.model_path.split('/')[-3]
        else:
            name_string = self.inf_conf.name

        self.use_ddp = self.inf_conf.use_ddp

        if self.use_ddp:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend='nccl')
            self.ddp_info = eu.get_ddp_info()
            self.rank_log(f"GPU {self.ddp_info['local_rank']} is connected", all=True)
            dt_string = [datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")]
            dist.broadcast_object_list(dt_string, src=0)
            self.output_dir = os.path.join(self.inf_conf.output_dir, name_string, dt_string[0], f"rank_{self.ddp_info['rank']}")
            self.num_replicas = self.ddp_info['world_size']
            self.rank = self.ddp_info['rank']
        else:
            dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
            self.output_dir = os.path.join(self.inf_conf.output_dir, name_string, dt_string)
            self.num_replicas = None
            self.rank = None
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.rank_log(f'Saving results to {self.output_dir}', all=True)
        
        if self.use_ddp:
            dist.barrier(device_ids=[self.ddp_info['local_rank']])

        self.rank_log(f'Loading model from {self.weights_path}')

        ckpt_pkl = eu.read_pkl(self.weights_path, use_torch=True)
       
        self.load_config(ckpt_pkl)
        self.tokenizer = self.build_tokenizer()
        self.model = self.build_model()
        if self.model_conf.get('use_peft', False):
            self.model = self.peft_model()
        self.model = self.wrap_model()

        eu.load_weights(self.model, ckpt_pkl)
        
        num_parameters = sum(p.numel() for p in self.model.parameters())
        self.rank_log(f'Number of model parameters : {num_parameters}')

        config_path = os.path.join(self.output_dir, 'inference_conf.yaml')
        self.rank_log(f'Saving inference config to {config_path}')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self.conf, f=f)



    def rank_log(self, msg, all=False):
        if not self.use_ddp:
            self.log.info(msg)
        else:
            if all:
                self.log.info(f"From rank { self.ddp_info['rank']}: {msg}")
            else:
                if self.ddp_info['rank'] in [0,-1]:
                    self.log.info(msg)

    
    def load_config(self, ckpt_pkl):
        conf = ckpt_pkl['conf']
        self.conf.model = conf.model 
        self.model_conf = self.conf.model
    

    def build_tokenizer(self):
        raise NotImplementedError("Subclasses must implement build_tokenizer")

    def build_model(self):
        raise NotImplementedError("Subclasses must implement build_model")
    
    def peft_model(self):
        raise NotImplementedError("Subclasses must implement peft_model")
    

    @property
    def data_loader(self):
        raise NotImplementedError("Subclasses must implement data_loader")


    def wrap_model(self):
        if not self.use_ddp:
            if torch.cuda.is_available():
                self.device = f"cuda:0"
            else:
                self.device = 'cpu'
            self.rank_log(f"Using device: {self.device}", all=True)
            return self.model.to(self.device)
        else:
            self.device = torch.device("cuda", self.ddp_info['local_rank'])
            model = self.model.to(self.device)
            self.rank_log(f"Multi-GPU inference on GPUs in DDP mode, node_id : {self.ddp_info['node_id']}")
            return model
       

    def run_inference(self):

        test_loader = self.create_test_loader()

        self.rank_log(f'Starting inference.')

        _ = self.inference_fn(test_loader)

        self.rank_log(f'Done.')




    def create_test_loader(self):
        
        test_dataset = self.data_loader.Dataset(self.data_conf, self.tokenizer, mode='test')
        
        test_sampler = self.data_loader.TestSampler(test_dataset, self.num_replicas, self.rank)
        
        test_loader = self.data_loader.DataLoader(
            test_dataset,
            self.tokenizer,
            sampler=test_sampler,
            batch_size=1,
            num_workers=self.inf_conf.num_loader_workers
        )

        return test_loader
    

    def inference_fn(self, test_loader):
        raise NotImplementedError("Subclasses must implement data_loader")


    def sample_tokens(self, logits):
        
        if self.temperature != 1.0:
            logits = logits / self.temperature

        probs = F.log_softmax(logits, dim=-1).exp()

        if self.sample_mode == 'greedy':
            tokens_ids = torch.argmax(probs, dim=-1)

        elif self.sample_mode == 'topk':
            top_k_probs, top_k_indices = torch.topk(probs, self.kp)
            dist = torch.distributions.Categorical(top_k_probs)
            selected_index = dist.sample()
            tokens_ids = torch.gather(top_k_indices, dim=2, index=selected_index.unsqueeze(-1)).squeeze(-1)

        elif self.sample_mode == 'topp':
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            top_p_mask = (cumulative_probs - sorted_probs) <= self.kp
            filtered_probs = sorted_probs * top_p_mask.float()
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

            dist=torch.distributions.Categorical(filtered_probs)
            selected_index = dist.sample()
            tokens_ids = torch.gather(sorted_indices, dim=2, index=selected_index.unsqueeze(-1)).squeeze(-1)

        else:
            raise ValueError(f'Invalid sample mode: {self.sample_mode}')
        
        tokens_probs = torch.gather(probs, dim=-1, index=tokens_ids.unsqueeze(-1)).squeeze(-1)

        return tokens_ids, tokens_probs
