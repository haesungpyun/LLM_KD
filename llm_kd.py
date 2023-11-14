from collections import defaultdict
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import math
import time
import torch
import pickle
from transformers import AutoTokenizer
from datasets import load_dataset
import evaluate
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
# from transformers.trainer_callback import DEFAULT_CALLBACKS, CallbackHandler
from transformers.integrations import get_reporting_integration_callbacks
from transformers.utils import find_labels
from transformers import AutoModelForSeq2SeqLM
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, LogitsProcessorList,
    MinLengthLogitsProcessor, StoppingCriteriaList, MaxLengthCriteria,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from logging import getLogger

from LLM_KD.data import DataWrapper
from LLM_KD.data_collator import Datacollator
from LLM_KD.optimizer_wrapper import OptimizerWrapper
from LLM_KD.scheduler_wrapper import SchedulerWrapper
from LLM_KD.utils import count_num_examples, prepare_input, get_model_param_count
# from LLM_KD.llama import llama


logger = getLogger(__name__)

if __name__ == "__main__":
    
    config = {
        'output_dir': 'my_awesome_model',
        'num_train_epochs': 3,
        'learning_rate': 2e-5,
        'train_batch_size': 16, 
        'valid_batch_size': 16, 
        'weight_decay': 0.01,
        'model_name': 't5-small'
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
    model = model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Load data_collator 
    data_collator = Datacollator(tokenizer=tokenizer, model=config["model_name"])
    
    # label name
    label_names = find_labels(model.__class__)

    # Data Wrapper
    data_wrapper = DataWrapper(
            config=config, model=model, tokenizer=tokenizer, 
            data_collator=data_collator, label_names=label_names
    )

    # Load train, test dataset 
    dataset = load_dataset("wmt14", "de-en")
    dataset['train'] = dataset['train'].select(range(1000))
    
    # Preprocess data
    tokenized  = dataset.map(data_wrapper.preprocess_function, batched=True)

    # Train, Valid, Test data
    train_data, valid_data, test_data = tokenized["train"], tokenized["validation"], tokenized["test"]
    

    # with open('tokenized_train.pkl', 'rb') as f:
    #     train_data = pickle.load(f)[:1000]
    
    # with open('tokenized_train.pkl', 'rb') as f:
    #     valid_data = pickle.load(f)

    # with open('tokenized_train.pkl', 'rb') as f:
    #     test_data = pickle.load(f)
    
    
    # Load Metric
    metric = evaluate.load("sacrebleu")

    # Data Collator
    default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
    data_collator = data_collator if data_collator is not None else default_collator

    # Optimizer, lr_scheduler
    optimizer, lr_scheduler = None, None

    # Callbacks
    # callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(None)
    # callback_handler = CallbackHandler(callbacks, model, tokenizer, optimizer, lr_scheduler)
    
    # Train dataloader
    train_dataloader = data_wrapper.get_train_dataloader(train_data)
    
    # Train batch size 
    total_train_batch_size = config.get("train_batch_size", 16) * config.get("gradient_accumulation_steps", 1) * config.get("world_size",1)
    
    # Number of update steps per epoch
    len_dataloader = len(train_dataloader)
    num_update_steps_per_epoch = max((len_dataloader // config.get("gradient_accumulation_steps", 1)), 1)
    num_examples = count_num_examples(config, train_dataloader)

    # Total number of training steps
    max_steps = math.ceil(config.get('num_train_epochs', 3.0) * num_update_steps_per_epoch)
    num_train_epochs = math.ceil(config.get('num_train_epochs', 3.0))
    num_train_samples = num_examples * config.get('num_train_epochs', 3.0)

    # Optimizer, lr_scheduler
    optimizer = OptimizerWrapper(model, config).create_optimizer()
    lr_scheduler = SchedulerWrapper(config, optimizer, num_training_steps=max_steps).create_scheduler()

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples:,}")
    logger.info(f"  Num Epochs = {num_train_epochs:,}")
    logger.info(f"  Instantaneous batch size per device = {config.get('per_device_train_batch_size', 8):,}")
    if config.get('per_device_train_batch_size', 8) != config.get('train_batch_size'):
        logger.info(f"  Training with DataParallel so batch size has been adjusted to: {config.get('train_batch_size'):,}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
    logger.info(f"  Gradient Accumulation steps = {config.get('gradient_accumulation_steps',1)}")
    logger.info(f"  Total optimization steps = {max_steps:,}")
    logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

    epochs_trained = 0  
    start_time = time.time()
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    total_batched_samples = 0
    tr_loss = torch.tensor(0.0)

    for epoch in range(epochs_trained, num_train_epochs):
        epoch_iterator = train_dataloader
        if hasattr(epoch_iterator, "set_epoch"):
            epoch_iterator.set_epoch(epoch)

        steps_in_epoch = (
            len(epoch_iterator)
            if len_dataloader is not None
            else config.get('max_steps', -1) * config.get("gradient_accumulation_steps",1)
        )
        
        step = -1
        for step, inputs in enumerate(epoch_iterator):
            total_batched_samples += 1
            
            model.train()
            inputs = prepare_input(config, inputs)

            labels = None
            
            outputs = model(**inputs)

            src_trg_pred = decode_for_llama(inputs, outputs, model, tokenizer, method='greedy')
            
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            tr_loss += loss        

            # Optimizer step
            optimizer.step()
            if not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step()

            model.zero_grad()

    logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")


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
        translation = tokenizer.batch_decode(greedy_output, skip_special_tokens=True)
    else:
        # get sequence length from label to decode only without label padding(i.e. -100) 
        # this is for the case that the last batch has shorter sequence length than the other batches
        seq_len = torch.count_nonzero(label, dim=-1)    

        translations = []
        for length, logits in zip(seq_len, outputs['logits']):
            logits = logits[:length].argmax(dim=-1)
            translations.append(tokenizer.batch_decode(logits, skip_special_tokens=True))
        
    return {"src": natural_src, "trg": gt_translation, "pred": translations}
