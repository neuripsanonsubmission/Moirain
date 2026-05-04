from collections import defaultdict
import torch
import torch.distributed as dist



class MetricTracker:
    def __init__(self, use_ddp=False):
        self.values = defaultdict(list)
        self.use_ddp = use_ddp

    def update(self, dict):
        for k,v in dict.items():
            self.values[k].append(v)

    def clear(self):
        self.values.clear()

    def get_log(self, eps = 1e-8):

        rolling_values = {}

        for key, entries in self.values.items():
            values = [v for v, _ in entries]
            samples = [s for _, s in entries if s is not None]

            # sum up values
            total_value = torch.stack(values).sum()

            # sum up samples if they exist, else None
            total_samples = torch.stack(samples).sum() if samples else None

            rolling_values[key] = (total_value, total_samples)

        if self.use_ddp:
            for key, (val, samp) in rolling_values.items():
                dist.all_reduce(val, op=dist.ReduceOp.SUM)
                if samp is not None:
                    dist.all_reduce(samp, op=dist.ReduceOp.SUM)
        
        output_dict = {}
        for key, (value, samples) in rolling_values.items():
            if samples is not None and samples != 0:
                output_dict[key] = (value / samples).cpu().numpy()
            else:
                output_dict[key] = (value).cpu().numpy()

        rolling_values.clear()
        del rolling_values

        tp = output_dict.pop("tp", None)
        tn = output_dict.pop("tn", None)
        fp = output_dict.pop("fp", None)
        fn = output_dict.pop("fn", None)

        # Compute only if all required values exist
        if all(v is not None for v in [tp, tn, fp, fn]):
            output_dict["accuracy"] = (tp + tn) / (tp + tn + fp + fn + eps)

        if all(v is not None for v in [tp, fp]):
            output_dict["precision"] = tp / (tp + fp + eps)

        if all(v is not None for v in [tp, fn]):
            output_dict["recall"] = tp / (tp + fn + eps)

        if all(v is not None for v in [tp, fp, fn]):
            output_dict["f1"] = (2 * tp) / (2 * tp + fp + fn + eps)
                    
        metric_log = ' '.join([f'{k}={v:.4f}' for k,v in output_dict.items()])

        output_dict.clear()
        del output_dict

        return metric_log