from typing import Any, List, Tuple
import torch

from .utils import ExplicitEnum, strtobool


class OptimizerType(ExplicitEnum):
        ADAMW_HF = "adamw_hf"
        ADAMW_TORCH = 'adamw_torch'
        ADAMW_TORCH_FUSED = "adamw_torch_fused"
        ADAMW_TORCH_XLA = "adamw_torch_xla"
        ADAMW_TORCH_NPU_FUSED = "adamw_torch_npu_fused"
        ADAMW_APEX_FUSED = "adamw_apex_fused"
        ADAFACTOR = "adafactor"
        ADAMW_ANYPRECISION = "adamw_anyprecision"
        SGD = "sgd"
        ADAGRAD = "adagrad"
        ADAMW_BNB = "adamw_bnb_8bit"
        ADAMW_8BIT = "adamw_8bit"  # just an alias for adamw_bnb_8bit
        LION_8BIT = "lion_8bit"
        LION = "lion_32bit"
        PAGED_ADAMW = "paged_adamw_32bit"
        PAGED_ADAMW_8BIT = "paged_adamw_8bit"
        PAGED_LION = "paged_lion_32bit"
        PAGED_LION_8BIT = "paged_lion_8bit"
        RMSPROP = "rmsprop"


class OptimizerWrapper:

    def __init__(
        self,
        model,
        config,
    ):
        self.config = config
        self.model = model

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        decay_parameters = self.get_decay_parameter_names()
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.config.get('weight_decay',0.0)
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs()

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return optimizer

    def get_decay_parameter_names(self) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        decay_parameters = self.get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters

    def get_parameter_names(self, model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in self.get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(self.model._parameters.keys())
        return result

    def get_optimizer_cls_and_kwargs(self) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """

        # parse args.optim_args
        optim_args = {}
        if self.config.get('optim_args'):
            for mapping in self.config.get('optim_args').replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": self.config.get('learning_rate', 5e-5)}

        adam_kwargs = {
            "betas": (self.config.get('adam_beta1', 0.9), self.config.get('adam_beta2', 0.999)),
            "eps": self.config.get('adam_epsilon', 1e-8),
        }
        if self.config.get('optim', 'adamw_torch') == OptimizerType.ADAFACTOR:
            from transformers.optimization import Adafactor

            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif self.config.get('optim', 'adamw_torch') == OptimizerType.ADAMW_HF:
            from transformers.optimization import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif self.config.get('optim', 'adamw_torch') in [OptimizerType.ADAMW_TORCH, OptimizerType.ADAMW_TORCH_FUSED]:
            from torch.optim import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
            if self.config.get('optim', 'adamw_torch') == OptimizerType.ADAMW_TORCH_FUSED:
                optimizer_kwargs.update({"fused": True})
        elif self.config.get('optim', 'adamw_torch') == OptimizerType.ADAMW_TORCH_XLA:
            try:
                from torch_xla.amp.syncfree import AdamW

                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
        elif self.config.get('optim', 'adamw_torch') == OptimizerType.ADAMW_TORCH_NPU_FUSED:
            try:
                from torch_npu.optim import NpuFusedAdamW

                optimizer_cls = NpuFusedAdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import FusedAdamW from torch_npu.")
        elif self.config.get('optim', 'adamw_torch') == OptimizerType.ADAMW_APEX_FUSED:
            try:
                from apex.optimizers import FusedAdam

                optimizer_cls = FusedAdam
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate apex FusedAdam but apex is not installed!")
        elif self.config.get('optim', 'adamw_torch') in [
            OptimizerType.ADAMW_BNB,
            OptimizerType.ADAMW_8BIT,
            OptimizerType.PAGED_ADAMW,
            OptimizerType.PAGED_ADAMW_8BIT,
            OptimizerType.LION,
            OptimizerType.LION_8BIT,
            OptimizerType.PAGED_LION,
            OptimizerType.PAGED_LION_8BIT,
        ]:
            try:
                from bitsandbytes.optim import AdamW, Lion

                is_paged = False
                optim_bits = 32
                optimizer_cls = None
                additional_optim_kwargs = adam_kwargs
                if "paged" in self.config.get('optim', 'adamw_torch'):
                    is_paged = True
                if "8bit" in self.config.get('optim', 'adamw_torch'):
                    optim_bits = 8
                if "adam" in self.config.get('optim', 'adamw_torch'):
                    optimizer_cls = AdamW
                elif "lion" in self.config.get('optim', 'adamw_torch'):
                    optimizer_cls = Lion
                    additional_optim_kwargs = {"betas": (self.config.get('adam_beta1', 0.9), self.config.get('adam_beta2', 0.999))}

                bnb_kwargs = {"is_paged": is_paged, "optim_bits": optim_bits}
                optimizer_kwargs.update(additional_optim_kwargs)
                optimizer_kwargs.update(bnb_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate bnb optimizer but bnb is not installed!")
            
        elif self.config.get('optim', 'adamw_torch') == OptimizerType.ADAMW_ANYPRECISION:
            try:
                from torchdistx.optimizers import AnyPrecisionAdamW

                optimizer_cls = AnyPrecisionAdamW
                optimizer_kwargs.update(adam_kwargs)

                # TODO Change dtypes back to M=FP32, Var = BF16, Kahan = False once they can be cast together in torchdistx.
                optimizer_kwargs.update(
                    {
                        "use_kahan_summation": strtobool(optim_args.get("use_kahan_summation", "False")),
                        "momentum_dtype": getattr(torch, optim_args.get("momentum_dtype", "float32")),
                        "variance_dtype": getattr(torch, optim_args.get("variance_dtype", "float32")),
                        "compensation_buffer_dtype": getattr(
                            torch, optim_args.get("compensation_buffer_dtype", "bfloat16")
                        ),
                    }
                )
            except ImportError:
                raise ValueError("Please install https://github.com/pytorch/torchdistx")
        elif self.config.get('optim', 'adamw_torch') == OptimizerType.SGD:
            optimizer_cls = torch.optim.SGD
        elif self.config.get('optim', 'adamw_torch') == OptimizerType.ADAGRAD:
            optimizer_cls = torch.optim.Adagrad
        elif self.config.get('optim', 'adamw_torch') == OptimizerType.RMSPROP:
            optimizer_cls = torch.optim.RMSprop
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {self.config.get('optim', 'adamw_torch')}")
        return optimizer_cls, optimizer_kwargs

