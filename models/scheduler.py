import math
from torch.optim.lr_scheduler import LRScheduler

class Scheduler(LRScheduler):
    def __init__(self, optimizer, sched_conf, last_epoch=-1):

        self.conf = sched_conf
        self.schedule_type = sched_conf.schedule_type
        self.warmup = sched_conf.warmup_steps is not None
        self.warmup_steps = sched_conf.warmup_steps

        self.start_step = last_epoch

        super().__init__(optimizer, last_epoch)
        

    def get_lr(self):

        step = self.last_epoch

        if self.warmup and step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        
        elif self.schedule_type == "cooldown":
            step -= self.start_step
            cooldown_steps = self.conf.cooldown_steps or 10000
            return [max(base_lr * (1-step / cooldown_steps), 0.0) for base_lr in self.base_lrs]

        else:

            if self.warmup:
                step -= self.warmup_steps
            
            if self.schedule_type == "constant":
                return [base_lr for base_lr in self.base_lrs]

            elif self.schedule_type == "gamma_decay":
                gamma = self.conf.gamma or 0.99
                period = self.conf.period or 10000
                return [base_lr * gamma ** (step // period) for base_lr in self.base_lrs]

            elif self.schedule_type == "cosin":
                period = self.conf. period or 10000
                return [base_lr * 0.5 * (1.0 + math.cos(float(step) / float(period) * math.pi)) for base_lr in self.base_lrs]

            elif self.schedule_type == "cosin_anneal":
                period = self.conf.period or 10000
                min_lr_scale = self.conf.min_lr_scale or 0.01
                return [min_lr_scale * base_lr + (base_lr - min_lr_scale * base_lr) * 0.5 * (1 + math.cos(float(step) / float(period) * math.pi)) for base_lr in self.base_lrs]

            elif self.schedule_type == "cosin_decay":
                period = self.conf.period or 200000
                min_lr_scale = self.conf.min_lr_scale or 0.1
                if step < period:
                    return [min_lr_scale * base_lr + (base_lr - min_lr_scale * base_lr) * 0.5 * (1 + math.cos(float(step) / float(period) * math.pi)) for base_lr in self.base_lrs]
                else:
                    return [min_lr_scale * base_lr for base_lr in self.base_lrs]
                
            else:
                raise ValueError(f"Unknown schedule: {self.schedule_type}")


    def state_dict(self):
        state = super().state_dict()
        state.pop('conf', None)
        state.pop('schedule_type', None)
        state.pop('warmup', None)
        state.pop('warmup_steps', None)
        state.pop('start_step', None)
        return state


    def load_state_dict(self, state_dict):
        state_dict.pop('conf', None)
        state_dict.pop('schedule_type', None)
        state_dict.pop('warmup', None)
        state_dict.pop('warmup_steps', None)
        state_dict.pop('start_step', None)
        super().load_state_dict(state_dict)
        self.start_step = self.last_epoch