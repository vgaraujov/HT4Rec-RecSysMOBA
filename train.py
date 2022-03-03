import torch
import logging
import os
import torch.nn.functional as F
import utils
import numpy as np

## Get the same logger from main"
logger = logging.getLogger("recdota")

def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))

def train(step, experiment, model, data_loader, loss_fn, device, optimizer, epoch, log_interval):
    with experiment.train():
        model.train()
        total_loss = 0
        total_acc = 0
        for batch_idx, [champs, items, target, attn_mask] in enumerate(data_loader):
            optimizer.zero_grad()
            champs = champs.to(device)
            items = items.to(device)
            target = target.to(device)
            batch, seq_len = target.shape
            target = target.view(-1)
            output = model(champs, items) # TODO: reshape items according to dataset
            # remove -100
            np_targets = target.cpu().detach().numpy()
            indices = np.where(np_targets == -100)[0].tolist()
            target = utils.delete(target, indices, 0)
            output = utils.delete(output, indices, 0)
            logit_sampled = output[:, target]
            # compute loss
            loss=loss_fn(logit_sampled)
            loss.backward()
            optimizer.step()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            acc = 1.*pred.eq(target.view_as(pred)).sum().item()/int(batch*seq_len)
            
            total_loss += loss.detach().item()
            total_acc += acc
            step += 1
            
            if experiment:
                experiment.log_metrics({'loss': total_loss/(batch_idx+1),
                                        'acc': total_acc/(batch_idx+1)},
                                        step = step)
            
            if batch_idx % log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(champs), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), acc, loss.detach().item()))
        
        # average loss # average acc
        final_acc = total_acc/len(data_loader)
        final_loss = total_loss/len(data_loader)
    
    return final_acc, final_loss, step
