import pickle
import os

import io
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_ddp_info():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    node_id = rank // world_size
    return {"node_id": node_id, "local_rank": local_rank, "rank": rank, "world_size": world_size}


class CPU_Unpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.

    https://github.com/pytorch/pytorch/issues/16797
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def write_pkl(save_path, pkl_data, create_dir=False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, 'wb') as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path, verbose=True, use_torch=False, map_location='cpu'):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location, weights_only=False)
        else:
            with open(read_path, 'rb') as handle:
                return pickle.load(handle)
    except Exception as e:
        try:
            with open(read_path, 'rb') as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                print(f'Failed to read {read_path}. First error: {e}\n Second error: {e2}')
            raise(e)


def write_checkpoint(
        ckpt_path,
        model_dict,
        conf,
        optimizer,
        scheduler,
        epoch,
        step,
        use_torch=True,
    ):
    
    checkpoint = dict(model_dict)

    checkpoint.update({
        'conf': conf,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'epoch': epoch,
        'step': step,
    })

    write_pkl(ckpt_path, checkpoint, use_torch=use_torch)



def load_weights(model, ckpt_pkl, key='model', strict=True):

    ckpt_model = {k.replace('module.', ''):v for k,v in ckpt_pkl[key].items()}

    if isinstance(model, DDP):
        result = model.module.load_state_dict(ckpt_model, strict=strict)
    else:
        result = model.load_state_dict(ckpt_model, strict=strict)

    if len(set(result.unexpected_keys)) != 0:
        print(set(result.unexpected_keys))
        raise ValueError(f"Unexpected keys in the {key} warm up checkpoint!")

    del ckpt_model
    torch.cuda.empty_cache()
