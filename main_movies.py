## Utilities
try:
    import comet_ml
    has_comet = True
except (ImportError):
    has_comet = False
import time
import os
import logging
import yaml
from timeit import default_timer as timer

## Libraries
import numpy as np
from box import box_from_file
from pathlib import Path

## Torch
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim

## Custom Imports
from logger import setup_logs
from seed import set_seed
from train import train, snapshot
from validation import validation
from dataset import MoviesDataset, DataCollatorForDota
from model import HTransformer
import losses

############ Control Center and Hyperparameter ###############
config = box_from_file(Path('config.yaml'), file_type='yaml')

# Override config
config.dataset.train_data_path = 'data/1m.pkl'
config.training.epochs = 150
config.model.te = False

if config.training.resume_name:
    run_name = config.training.resume_name
else:
    run_name = config.model.model_type + time.strftime("-%Y-%m-%d_%H_%M_%S")

# setup logger    
global_timer = timer() # global timer
logger = setup_logs(config.training.logging_dir, run_name) # setup logs
logger.info('### Experiment {} ###'.format(run_name))
logger.info('### Hyperparameter summary below ###\n {}'.format(config))
# setup of comet_ml
if has_comet:
    logger.info('### Logging with comet_ml ###')
    if config.comet.previous_experiment:
        logger.info('===> using existing experiment: {}'.format(config.comet.previous_experiment))
        experiment = comet_ml.ExistingExperiment(api_key=config.comet.api_key,
                                                 previous_experiment=config.comet.previous_experiment)    
    else:
        logger.info('===> starting new experiment')
        experiment = comet_ml.Experiment(api_key=config.comet.api_key,
                                         project_name="recdota")
    experiment.set_name(run_name)
    experiment.log_parameters({**config.training.to_dict() , 
                               **config.dataset.to_dict() , 
                               **config.model.to_dict()})
else:
    experiment = None
    
# define if gpu or cpu
use_cuda = not config.training.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
logger.info('===> use_cuda is {}'.format(use_cuda))
# set seed for reproducibility
set_seed(config.training.seed, use_cuda)

## Loading the dataset
logger.info('===> loading train and validation dataset')
dataset = MoviesDataset(config)

data_collator = DataCollatorForDota(max_length = config.dataset.max_seq_length)

# split to train val
validation_split = 0.2 # 20% of dataset for validation
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))    
# get random indices (shuffle equivalent)
train_indices, valid_indices = indices[split:], indices[:split]
np.random.shuffle(train_indices)
# create dataloader
train_sampler = data.sampler.SubsetRandomSampler(train_indices)
validation_sampler = data.sampler.SubsetRandomSampler(valid_indices)
    
multiplier = torch.cuda.device_count() if not config.training.no_cuda else 1
batch_size = int(config.training.batch_size*multiplier)
train_loader = data.DataLoader(dataset,
                               batch_size=batch_size,
                               collate_fn=data_collator,
                               sampler=validation_sampler,
                               drop_last=True
                              )
validation_loader = data.DataLoader(dataset, 
                                    batch_size=batch_size,
                                    collate_fn=data_collator,
                                    sampler=validation_sampler,
                                    drop_last=True
                                   )

config.dataset.n_items = len(dataset.item2id)
config.dataset.n_champs = len(dataset.user2id)

# create a model
model = HTransformer(config=config)

# load model if resume mode
if config.training.resume_name:
    logger.info('===> loading a checkpoint')
    checkpoint = torch.load('{}/{}-{}'.format(config.training.logging_dir, run_name, 'model_best.pth'))
    model.load_state_dict(checkpoint['state_dict'])
# line for multi-gpu
if config.training.multigpu and torch.cuda.device_count() > 1:
    logger.info("===> let's use {} GPUs!".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
# move to device
model.to(device)

# Adam optimizer with a learning rate of 2e-4
optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=2e-4, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True)
if config.training.resume_name:
    optimizer.load_state_dict(checkpoint['optimizer'])

# create loss function
loss_fn = losses.LossFunction(loss_type=config.model.loss_fn)
    
model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info('### Model summary below ###\n {}'.format(str(model)))
logger.info('===> Model total parameter: {}\n'.format(model_params))
# if has_comet: experiment.set_model_graph(str(model))

## Add tags for details
if has_comet:
    experiment.add_tags([config.model.model_type,
                        config.model.emb_fusion,
                        config.dataset.max_seq_length,
                        config.model.loss_fn,
                        "dim="+str(config.model.emb_dim),
                        "layer1="+str(config.model.layers1),
                        "layer2="+str(config.model.layers2),
                        "head="+str(config.model.n_heads),
                        "user="+str(config.model.ue),
                        "pe="+str(config.model.pe),
                        "movies"])

## Start training 
if config.training.resume_name:
    best_acc = checkpoint['validation_acc']
    best_loss = checkpoint['validation_loss']
    best_epoch = checkpoint['epoch']
    step = checkpoint['step_train']
    initial_epoch = checkpoint['epoch']
else:
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1 
    step = 0
    initial_epoch = 1
    
logger.info('### Training begins at epoch {} and step {} ###'.format(initial_epoch,step))
for epoch in range(initial_epoch, config.training.epochs + 1):
    epoch_timer = timer()
    # Train and validate
    tr_acc, tr_loss, step = train(
        step, 
        experiment, 
        model, 
        train_loader, 
        loss_fn, 
        device, 
        optimizer, 
        epoch, 
        config.training.log_interval)
    
    if not epoch % 10:    
        val_acc, val_loss = validation(
            step, 
            experiment, 
            model, 
            validation_loader, 
            loss_fn, 
            device)
        # Save
        if val_loss < best_loss: 
            best_loss = min(val_loss, best_loss)
            if torch.cuda.device_count() > 1 and not config.training.no_cuda:
                dict_to_save = model.module.state_dict()
            else:
                dict_to_save = model.state_dict()
            snapshot(config.training.logging_dir, run_name, {
                'epoch': epoch,
                'step_train': step,
                'validation_acc': val_acc,
                'validation_loss': val_loss,
                'state_dict': dict_to_save,
                'optimizer': optimizer.state_dict(),
            })
            best_epoch = epoch

    end_epoch_timer = timer()
    logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, config.training.epochs, end_epoch_timer - epoch_timer))
    
## end 
end_global_timer = timer()
logger.info("################## Success #########################")
logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))
