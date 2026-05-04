import torch
import numpy as np
import pandas as pd
import gemmi
import h5py



class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_conf, tokenizer, mode):

        self.tokenizer = tokenizer
        self.mode = mode
        self.data_conf = data_conf
        self.data_format = data_conf.data_format

        # Set paths and max_len based on mode
        if self.mode == 'train':
            self.max_len = self.data_conf.max_len
            csv_path = self.data_conf.csv_path_train
            data_path = self.data_conf.data_path_train
        elif self.mode == 'validation':
            self.max_len = self.data_conf.max_len
            csv_path = self.data_conf.csv_path_val
            data_path = self.data_conf.data_path_val
        elif self.mode == 'test':
            csv_path = self.data_conf.csv_path_test
            data_path = self.data_conf.data_path_test
        else:
            raise ValueError(f'Invalid dataset mode: {self.mode}')

        # Load data based on format
        if self.data_format == 'csv':
            self.csv = pd.read_csv(csv_path)
            self.data_path = data_path

        elif self.data_format == 'h5':
            file = h5py.File(data_path, "r")
            self.data = file["sequences"]

        elif self.data_format == 'fasta':
            with open(data_path) as f:
                fasta_str = f.read()
            self.data = gemmi.read_pir_or_fasta(fasta_str)

        elif self.data_format == 'base':
            self.size = self.data_conf.size

        else:
            raise ValueError(f'Invalid data format: {self.data_format}')
        
    
    def __getitem__(self, example_idx):
        
        if self.data_format == 'csv':

            csv_row = self.csv.iloc[example_idx]
            id = csv_row['id']
            feats = self.produce_sample(csv_row)
            return feats, id

        elif self.data_format == 'h5':

            entry = self.data[example_idx]
            id = entry['name'].decode('utf-8')
            feats = self.produce_sample(entry['sequence'].decode('utf-8'))
            return feats, id
        
        elif self.data_format == 'fasta':
            
            elt = self.data[example_idx]
            id = elt.header
            feats = self.produce_sample(elt.seq)
            return feats, id

        elif self.data_format == 'base':

            feats = self.produce_sample()
            return feats, str(example_idx)

        else:
            raise ValueError(f'Invalid data format: {self.data_format}')

    
    def produce_sample(self, *args, **kwargs):

        if self.mode == 'train':
            feats = self.produce_sample_train(*args, **kwargs)
        elif self.mode == 'validation':
            feats = self.produce_sample_validation(*args, **kwargs)
        elif self.mode == 'test':
            feats = self.produce_sample_test(*args, **kwargs)
        else:
            raise ValueError(f'Invalid dataset mode: {self.mode}')
        
        return feats
    
    
    def produce_sample_train(self):
        raise NotImplementedError("Subclasses must implement produce_sample_train")
    
    def produce_sample_validation(self, *args, **kwargs):
        return self.produce_sample_train(*args, **kwargs)
    
    def produce_sample_test(self):
        raise NotImplementedError("Subclasses must implement produce_sample_test")
    
    
    def __len__(self):
        if hasattr(self, 'csv') and self.csv is not None:
            return len(self.csv)
        elif hasattr(self, 'data') and self.data is not None:
            return len(self.data)
        elif hasattr(self, 'size') and self.size is not None:
            return self.size
        else:
            raise NotImplementedError("I don't know the size of data!")
    
    
    def __del__(self):
        if hasattr(self, 'data') and self.data is not None:
            try:
                self.data.close()
            except:
                pass


class Sampler(torch.utils.data.Sampler):

    def __init__(self, data_conf, dataset, num_replicas=None, rank=None):

        self.data_conf = data_conf
        self.dataset = dataset
        self.sample_mode = data_conf.sample_mode

        self.epoch = 0
        self.start_idx = None
        self.end_idx = None

        self.num_replicas = num_replicas
        self.rank = rank

        if self.sample_mode == 'normal':
            self.sampler_len = len(self.dataset)
        elif self.sample_mode == 'cluster_batch':
            self.clust_key = data_conf.clust_key
            self.sample_num = data_conf.sample_num
            self.csv = self.dataset.csv
            self.sampler_len = self.sample_num*len(set(self.csv[self.clust_key].values))
        else:
            raise ValueError(f'Invalid sample mode: {self.sample_mode}')

        if self.num_replicas is not None:
            self.sampler_len = np.ceil(self.sampler_len / self.num_replicas).astype(int)
            self.total_size = self.sampler_len * self.num_replicas
        

    def __iter__(self) :

        if self.sample_mode == 'normal':
            sampled_indices = list(range(len(self.dataset)))
        elif self.sample_mode == 'cluster_batch':
            sampled_clusters = self.csv.groupby(self.clust_key).sample(self.sample_num, random_state=self.epoch, replace=True)
            sampled_indices = sampled_clusters.index.tolist()
        else:
            raise ValueError(f'Invalid sample mode: {self.sample_mode}')

        g = torch.Generator()
        g.manual_seed(self.epoch)
        perm = torch.randperm(len(sampled_indices), generator=g).tolist()
        sampled_indices = [sampled_indices[i] for i in perm]

        if self.rank is not None:
            padding_size = self.total_size - len(sampled_indices)
            sampled_indices = sampled_indices + sampled_indices[:padding_size]
            
            assert len(sampled_indices) == self.total_size

            sampled_indices = sampled_indices[self.rank:self.total_size:self.num_replicas]
            
            assert len(sampled_indices) == self.sampler_len

        if self.start_idx is not None:
            idx_mod = self.start_idx % self.sampler_len
            sampled_indices = sampled_indices[idx_mod:]
            self.start_idx = None

        if self.end_idx is not None:
            idx_mod = min(self.end_idx, self.sampler_len)
            sampled_indices = sampled_indices[:idx_mod]

        return iter(sampled_indices)
    

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_start(self, idx):
        self.start_idx = idx

    def set_end(self, idx):
        self.end_idx = idx

    def __len__(self):
        return self.sampler_len




class TestSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None):
        
        self.dataset = dataset

        self.num_replicas = num_replicas
        self.rank = rank

        if self.rank is not None:
            self.total_size = len(self.dataset)
            self.sampler_len = np.ceil(self.total_size / self.num_replicas).astype(int)

            self.start = self.sampler_len * self.rank
            self.end = min(self.start + self.sampler_len, self.total_size)
        else:
            self.sampler_len = len(self.dataset)

    
    def __iter__(self):

        if self.rank is None:
            indices = list(range(len(self.dataset)))
        else:
            indices = iter(range(self.start, self.end))

        return iter(indices)

    def __len__(self):
        return self.sampler_len




    


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, 
                dataset,
                tokenizer,
                sampler,
                batch_size,
                num_workers=0,
                prefetch_factor=2
        ):

        self.tokenizer = tokenizer

        prefetch_factor = None if num_workers == 0 else prefetch_factor

        super().__init__(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=self.cat_features,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
            pin_memory=False,
            drop_last=False,
            multiprocessing_context='fork' if num_workers != 0 else None,
            )

    def cat_features(self, batch):
        raise NotImplementedError("Subclasses must implement collate_fn")



 