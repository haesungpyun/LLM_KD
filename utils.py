import random
import numpy as np
from enum import Enum
from typing import Any, Mapping, Union
import torch
from torch.utils.data import DataLoader

from transformers.trainer_pt_utils import IterableDatasetShard

from transformers import LogitsProcessorList, MinLengthLogitsProcessor
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in {"y", "yes", "t", "true", "on", "1"}:
        return 1
    if val in {"n", "no", "f", "false", "off", "0"}:
        return 0
    raise ValueError(f"invalid truth value {val!r}")

def count_num_examples(config, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            dataset = dataloader.dataset
            # Special case for IterableDatasetShard, we need to dig deeper
            if isinstance(dataset, IterableDatasetShard):
                return len(dataloader.dataset.dataset)
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * config.get('per_device_train_batch_size', 8)

def prepare_input(config, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: prepare_input(config, v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(prepare_input(config, v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))}
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data

def get_model_param_count(model, trainable_only=False):
    """
    Calculate model's total param count. If trainable_only is True then count only those requiring grads
    """
    def numel(p):
        return p.numel()

    return sum(numel(p) for p in model.parameters() if not trainable_only or p.requires_grad)


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available

def seed_worker(seed: int = None):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        seed = torch.initial_seed() % 2**32
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def decode_for_llama(inputs, outputs, model, tokenizer, method='greedy'):
    # natural language source sentence 
    natural_src = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)

    # ground truth translation
    label = inputs['labels'].clone()
    label = torch.where(label != -100, label, tokenizer.pad_token_id)
    gt_translation = tokenizer.batch_decode(label, skip_special_tokens=True)
    
    if method == 'greedy':
        # For Greedy decoding, we have to forward the model without teacher forcing.
        # So, we have to prepare the decoder input ids. 
        # define decoder start token ids
        decoder_input = torch.ones((inputs['input_ids'].size(0), 1), device=model.device, dtype=torch.long)
        decoder_input = decoder_input * model.config.decoder_start_token_id

        # add encoder_outputs to model keyword arguments
        model_kwargs = {
            "encoder_outputs": BaseModelOutputWithPastAndCrossAttentions(outputs['encoder_last_hidden_state']),
            "attention_mask": inputs['attention_mask']
        }
        # instantiate logits processors
        logits_processor = LogitsProcessorList(
            [ MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),]
        )            

        greedy_output = model.greedy_search(decoder_input, logits_processor=logits_processor, **model_kwargs)
        translations = tokenizer.batch_decode(greedy_output, skip_special_tokens=True)
    else:
        # get sequence length from label to decode only without label padding(i.e. -100) 
        # this is for the case that the last batch has shorter sequence length than the other batches
        seq_len = torch.count_nonzero(label, dim=-1)    

        translations = []
        for length, logits in zip(seq_len, outputs['logits']):
            logits = logits[:length].argmax(dim=-1)
            translations.append(tokenizer.batch_decode(logits, skip_special_tokens=True))
        
    return {"src": natural_src, "trg": gt_translation, "pred": translations}
