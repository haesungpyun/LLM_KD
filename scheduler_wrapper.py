import math
from typing import Optional, Union

import torch
from torch.optim import Optimizer

from .schedule_functions import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

class SchedulerWrapper:
    def __init__(
        self,
        config,
        optimizer: Optional[Optimizer] = None,
        num_training_steps: Optional[int] = None,
    ):
        self.config = config
        self.optimizer = optimizer
        self.num_training_steps = num_training_steps


    def create_scheduler(self):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        lr_scheduler = self.get_scheduler(
            self.config.get('lr_scheduler_type', 'linear'),
            num_warmup_steps=self.get_warmup_steps()
        )
        return lr_scheduler

    def get_scheduler(
        self,
        name: Union[str, SchedulerType],
        num_warmup_steps: Optional[int] = None,
        scheduler_specific_kwargs: Optional[dict] = None,
    ):
        """
        Unified API to get any scheduler from its name.

        Args:
            name (`str` or `SchedulerWrapper`):
                The name of the scheduler to use.
            optimizer (`torch.optim.Optimizer`):
                The optimizer that will be used during training.
            num_warmup_steps (`int`, *optional*):
                The number of warmup steps to do. This is not required by all schedulers (hence the argument being
                optional), the function will raise an error if it's unset and the scheduler type requires it.
            num_training_steps (`int``, *optional*):
                The number of training steps to do. This is not required by all schedulers (hence the argument being
                optional), the function will raise an error if it's unset and the scheduler type requires it.
            scheduler_specific_kwargs (`dict`, *optional*):
                Extra parameters for schedulers such as cosine with restarts. Mismatched scheduler types and scheduler
                parameters will cause the scheduler function to raise a TypeError.
        """
        name = SchedulerType(name)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
        if name == SchedulerType.CONSTANT or name == SchedulerType.REDUCE_ON_PLATEAU:
            return schedule_func(self.optimizer)

        # All other schedulers require `num_warmup_steps`
        if num_warmup_steps is None:
            raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

        if name == SchedulerType.CONSTANT_WITH_WARMUP:
            return schedule_func(self.optimizer, num_warmup_steps=num_warmup_steps)

        if name == SchedulerType.INVERSE_SQRT:
            return schedule_func(self.optimizer, num_warmup_steps=num_warmup_steps)

        # All other schedulers require `num_training_steps`
        if self.num_training_steps is None:
            raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

        if scheduler_specific_kwargs is None:
            scheduler_specific_kwargs = {}

        return schedule_func(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_training_steps,
            **scheduler_specific_kwargs,
        )

    def get_warmup_steps(
        self, 
    ):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.config.get('warmup_steps', 0) 
            if self.config.get('warmup_steps', 0) > 0 
            else math.ceil(self.num_training_steps * self.config.get('warmup_ratio', 0.0))
        )
        return warmup_steps
    

