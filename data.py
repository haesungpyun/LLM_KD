import random
import numpy as np
from typing import Callable, Optional, List
import inspect
import torch
from torch.utils.data import Dataset, DataLoader
from packaging import version

import datasets
import transformers.utils.logging as logging
from transformers.trainer_utils import RemoveColumnsCollator
from transformers import PreTrainedTokenizerBase
from transformers.utils import is_datasets_available
from transformers.trainer_pt_utils import LengthGroupedSampler, RandomSampler
from transformers.data.data_collator import DataCollator

from .utils import set_seed, seed_worker

logger = logging.get_logger(__name__)

class DataWrapper:
    def __init__(
        self,
        config,
        model,
        tokenizer,
        data_collator,
        label_names = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.label_names = label_names

    def preprocess_function(
        self,
        examples   
    ):
        source_lang = 'en'
        target_lang = 'de'
        prefix = f"translate English to German: "
        
        logger.info(f'prefix: , {prefix}')
        
        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = self.tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        
        return model_inputs

    def get_train_dataloader(
        self,
        train_dataset: Dataset,
    )-> DataLoader:
                            
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            self.data_collator = self._get_collator_with_removed_columns(self.data_collator, description="training")

        dataloader_params = {
            "batch_size": self.config.get("train_batch_size"),
            "collate_fn": self.data_collator,
            "num_workers": self.config.get("dataloader_num_workers", 0),
            "pin_memory": self.config.get("dataloader_pin_memory", True),
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler(train_dataset)
            dataloader_params["drop_last"] = self.config.get("dataloader_drop_last", False)
            dataloader_params["worker_init_fn"] = seed_worker

        return DataLoader(train_dataset, **dataloader_params)

    def _remove_unused_columns(
        self,
        dataset: datasets.Dataset, 
        description: Optional[str] = None,
        remove_unused_columns: Optional[bool] = True,
    ):      

        if not remove_unused_columns:
            return dataset
        signature_columns = self._set_signature_columns_if_needed()

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _set_signature_columns_if_needed(
        self,
        _signature_columns=None,        
    ):
        if _signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            _signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            _signature_columns += list(set(["label", "label_ids"] + self.label_names))
        return _signature_columns

    def _get_collator_with_removed_columns(
        self,
        description: Optional[str] = None,
        remove_unused_columns: Optional[bool] = True,
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not remove_unused_columns:
            return self.data_collator
        signature_columns = self._set_signature_columns_if_needed()

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=self.data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator
    
    def _get_train_sampler(
        self,
        train_dataset
    ):
        generator = torch.Generator()
        if self.config.get('data_seed') is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.config.get('data_seed')
        generator.manual_seed(seed)
        
        seed = self.config.get('data_seed') if self.config.get('data_seed') is not None else self.config.get('seed', 42)

        if self.config.get("group_by_length", False):
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                lengths = (
                    train_dataset[self.config.get('length_column_name', 'length')]
                    if self.config.get('length_column_name', 'length') in train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.config.get("train_batch_size") * self.config.get("gradient_accumulation_steps",1),
                dataset=train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else:
            return RandomSampler(train_dataset)
        
    def compute_metric(self, eval_pred, tokenizer, metric):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_labels = [label.strip() for label in decoded_labels]
        decoded_preds = [pred.strip() for pred in decoded_preds]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result