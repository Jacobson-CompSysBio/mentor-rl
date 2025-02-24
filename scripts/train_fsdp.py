# necessary imports
import os, sys, glob, time
from datetime import timedelta
import argparse
import tqdm.auto as tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, DeepSpeedPlugin, InitProcessGroupKwargs
from DGXutils import GetLowestGPU # only need for DGX

sys.path.append('../')

# custom imports 
from utils.dataset import CustomDataset
from utils.model import CustomModel
from utils.loss import CustomLoss

def main():
    ## CONFIG
    device = GetLowestGPU() # only need for DGX
    train_split = 0.8

    num_epochs = 10
    B = 32

    ## FSDP INIT
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        kwargs_handlers=[kwargs])
    
    ## LOAD DATA
    dataset = CustomDataset() # skeleton code for dataset rn

    # get train/test split
    train_idx = np.random.choice(len(dataset), int(train_split*len(dataset)), replace=False)
    val_idx = np.array(list(set(range(len(dataset))) - set(train_idx)))

    # subset
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    # dataloaders
    train_loader = DataLoader(train_dataset,
                            batch_size=B, 
                            shuffle=False) # shuffle to false because accelerate auto-shuffles

    val_loader = DataLoader(val_dataset, 
                            batch_size=B, 
                            shuffle=False)

    ## LOAD MODEL
    model = CustomModel() # skeleton code for model rn
    model = model.to(device)

    ## TRAIN LOOP

    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = CustomLoss() # skeleton code for loss rn
    train_pbar = tqdm.tqdm(range(num_epochs * len(train_loader)), disable=not accelerator.is_main_process)
    val_pbar = tqdm.tqdm(range(num_epochs * len(val_loader)), disable=not accelerator.is_main_process)
    best_loss = np.inf()
    best_loss_epoch = 0
    save_path = '../checkpoints/'
    log_path = '../logs/log.txt'
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')

    # prepare model, dl, opt for training
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # in case we're using GCN
    # for name, module in model._fp32_modules.items():
    #     model._fp32_modules[name] = module.to(accelerator.device)

    # enable grad checkpointing for additional mem savings
    if hasattr(model.model, "gradient_checkpointing_enable"):
        model.model.gradient_checkpointing_enable()
        accelerator.print("Enabled gradient checkpointing for HF model.")

    # loop through epochs
    for epoch in range(num_epochs):
        # instantiate loss tracking
        train_loss = 0.0
        val_loss = 0.0

        # backprop
        model.train()
        for batch in train_loader:
            # load batches + send to device
            x, y = batch
            x, y = x.to(device), y.to(device)

            # zero grad, forward, backward, step
            optimizer.zero_grad()
            y_pred = model(x)
            L = loss(y_pred, y)
            accelerator.backward(L)
            train_loss += L.item()
            optimizer.step()
            train_pbar.update(1)

        # average loss for epoch
        avg_train_loss = train_loss / len(train_loader)

        # validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                L = loss(y_pred, y)
                val_loss += L.item()
                val_pbar.update(1)
        avg_val_loss = val_loss / len(val_loader)

        # update best loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_loss_epoch = epoch
            torch.save(model.state_dict(), save_path + f'model_epoch{epoch}.pth') # save model if best 

        # update logs
        with open(log_path, 'a') as f:
            f.write(f'{epoch+1},{avg_train_loss},{avg_val_loss}\n')

        # print update
        accelerator.print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Best Val Loss: {best_loss:.4f} at epoch {epoch+1}')
        
        # print model inference
        if num_epochs % 5 == 0:
            accelerator.print(model.inference(x))

if __name__ == '__main__':
    main()