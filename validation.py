import numpy as np
import logging
import torch
import torch.nn.functional as F
import metrics
import utils

## Get the same logger from main"
logger = logging.getLogger("recdota")

def validation(step, experiment, model, data_loader, loss_fn, device):
    with experiment.validate():
        logger.info("Starting Validation")
        model.eval()
        total_loss = 0
        total_acc = 0
        total_precision1 = 0
        total_recall1 = 0
        total_map1 = 0
        total_mrr1 = 0
        total_precision3 = 0
        total_recall3 = 0
        total_map3 = 0
        total_mrr3 = 0
        total_precision5 = 0
        total_recall5 = 0
        total_map5 = 0
        total_mrr5 = 0
        total_precision10 = 0
        total_recall10 = 0
        total_map10 = 0
        total_mrr10 = 0
        with torch.no_grad():
            for batch_idx, [champs, items, target, attn_mask] in enumerate(data_loader):
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
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                acc = 1.*pred.eq(target.view_as(pred)).sum().item()/int(batch*seq_len)

                total_loss += loss.detach().item()
                total_acc += acc
                
                recall, precision, ap, mrr = metrics.evaluate(output, target, 1)
                total_precision1 += precision
                total_recall1 += recall
                total_map1 += ap
                total_mrr1 += mrr
                
                recall, precision, ap, mrr = metrics.evaluate(output, target, 3)
                total_precision3 += precision
                total_recall3 += recall
                total_map3 += ap
                total_mrr3 += mrr
                
                recall, precision, ap, mrr = metrics.evaluate(output, target, 5)
                total_precision5 += precision
                total_recall5 += recall
                total_map5 += ap
                total_mrr5 += mrr
                
                recall, precision, ap, mrr = metrics.evaluate(output, target, 10)
                total_precision10 += precision
                total_recall10 += recall
                total_map10 += ap
                total_mrr10 += mrr
        
        # average loss # average acc
        final_acc = total_acc/len(data_loader)
        final_loss = total_loss/len(data_loader)
        final_precision1 = total_precision1/len(data_loader)
        final_recall1 = total_recall1/len(data_loader)
        final_map1 = total_map1/len(data_loader)
        final_mrr1 = total_mrr1/len(data_loader)
        final_precision3 = total_precision3/len(data_loader)
        final_recall3 = total_recall3/len(data_loader)
        final_map3 = total_map3/len(data_loader)
        final_mrr3 = total_mrr3/len(data_loader)
        final_precision5 = total_precision5/len(data_loader)
        final_recall5 = total_recall5/len(data_loader)
        final_map5 = total_map5/len(data_loader)
        final_mrr5 = total_mrr5/len(data_loader)
        final_precision10 = total_precision10/len(data_loader)
        final_recall10 = total_recall10/len(data_loader)
        final_map10 = total_map10/len(data_loader)
        final_mrr10 = total_mrr10/len(data_loader)
        if experiment:
            experiment.log_metrics({'loss': final_loss,
                                    'acc': final_acc,
                                    'recall@1': final_recall1,
                                    'mrr@1': final_mrr1,
                                    'precision@1': final_precision1,
                                    'map@1': final_map1,
                                    'recall@3': final_recall3,
                                    'mrr@3': final_mrr3,
                                    'precision@3': final_precision3,
                                    'map@3': final_map3,
                                    'recall@5': final_recall5,
                                    'mrr@5': final_mrr5,
                                    'precision@5': final_precision5,
                                    'map@5': final_map5,
                                    'recall@10': final_recall10,
                                    'mrr@10': final_mrr10,
                                    'precision@10': final_precision10,
                                    'map@10': final_map10},
                                    step = step)
        logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                    final_loss, final_acc))
        
    return final_acc, final_loss
