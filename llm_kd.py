import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import math
import time
import torch
import pickle
import evaluate
from logging import getLogger
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers.data.data_collator import DataCollatorWithPadding, default_data_collator
# from transformers.trainer_callback import DEFAULT_CALLBACKS, CallbackHandler
from transformers.utils import find_labels
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from LLM_KD.data import DataWrapper
from LLM_KD.data_collator import Datacollator
from LLM_KD.optimizer_wrapper import OptimizerWrapper
from LLM_KD.scheduler_wrapper import SchedulerWrapper
from LLM_KD.utils import count_num_examples, prepare_input, get_model_param_count, decode_for_llama
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
            labels = prepare_input(config, inputs.get("labels", None))
            
            outputs = model(**inputs)
            
            src_trg_pred = decode_for_llama(inputs, outputs, model, tokenizer, method='plain')

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            # move labels to correct device to enable PP
            labels = labels.to(outputs['logits'].device)
            loss = loss_fct(outputs['logits'].view(-1, outputs['logits'].size(-1)), labels.view(-1))
                
            model.zero_grad()
            loss.mean().backward()
            # Optimizer step
            optimizer.step()
            
            if not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step()

            tr_loss += loss

    logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")

