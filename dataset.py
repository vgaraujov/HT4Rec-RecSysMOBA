""" Dataset loader for the dataset """
import torch
import torch.utils.data as data
import numpy as np
import os
import functools
import operator
import logging
import itertools
import random
import pickle
import pandas as pd

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict

## Get the same logger from main"
logger = logging.getLogger("recdota")


class DotaDataset(data.Dataset):
    def __init__(self, config, split):

        """
        Args:
            config (box): hyperparameters file
        """
        # Hyperparameters
        self.split = split
        if self.split == 'train':
            self.data_path = config.dataset.train_data_path
        else:
            self.data_path = config.dataset.test_data_path
        self.items_path = config.dataset.item_path
        self.champs_path = config.dataset.champ_path
        self.max_seq = config.dataset.max_seq_length
        
        # load paths of books in subfolder <data>
        logger.info("Loading features from dataset at %s", self.data_path)
        self.examples = self.open_dataset(self.data_path)
        self.id2item, self.item2id = self.open_items_info(self.items_path)
        self.id2champ, self.champ2id = self.open_champs_info(self.champs_path)

    def __getitem__(self, i):
        list_champs = list(self.examples[i].keys())
        champions = self.convert_to_ids(self.champ2id, list_champs)
        items, labels = self.convert_to_multihot(self.examples[i])
        return torch.tensor(champions, dtype=torch.long), items, torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)
    
    def open_dataset(self, path):
        pickle_in = open(path,"rb")
        data = pickle.load(pickle_in)
        pickle_in.close()
        return data
    
    def open_items_info(self, items_path):
        items_df = pd.read_csv(items_path)
        items_df.drop([39,146,122,36,42,157,137,40,38,41,44, 32, 85, 143, 
                       145, 147, 148, 149, 151, 154, 155, 158, 159, 160, 
                       161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 
                       171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 
                       181, 182, 183, 184, 185, 186, 187, 188], inplace=True)  
        items_df.reset_index(drop=True, inplace=True)
        dictionary = {i+1 : v for i, v in enumerate(items_df['item_id'].tolist())}
        dictionary[0] = 0
        reversed_dictionary = {value : key for (key, value) in dictionary.items()}
        return dictionary, reversed_dictionary
    
    def open_champs_info(self, champs_path):
        champs_df = pd.read_csv(champs_path)
        champs_df.drop([106, 111], inplace = True)
        champs_df.drop(['name'], axis = 1, inplace = True)
        champs_df.reset_index(drop=True, inplace=True)
        dictionary = {i+1 : v for i, v in enumerate(champs_df['hero_id'].tolist())}
        dictionary[0] = 0
        reversed_dictionary = {value : key for (key, value) in dictionary.items()}
        return dictionary, reversed_dictionary
    
    def convert_to_ids(self, map_dict, tokens):
        return [map_dict[int(i)] for i in tokens]
    
    def convert_to_multihot(self, sample):
        champs = list(sample.keys())
        actual_len = len(sample[champs[-1]])
        
        user_items = sample[champs[0]]
        user_items = list(itertools.chain(*user_items))
        labels = self.convert_to_ids(self.item2id, user_items[1:])
        
        list_one_hot = []
        for champ in champs:
            result = []
            item_sets = sample[champ]
            item_sets = item_sets[:actual_len]
            for i_set in item_sets:
                item_set = []
                for i in i_set:
                    item_set.append(self.item2id[int(i)]+1) # this is for shift
                result.append(item_set)
            list_one_hot.append(self.to_onehot(result, len(self.item2id), dtype=torch.float32))

        items_tensor = torch.stack(list_one_hot, dim=1) # shape S,C,D
        return items_tensor, labels

    def to_onehot(self, labels, n_categories, dtype=torch.float32):
        batch_size = len(labels)
        one_hot_labels = torch.zeros(size=(batch_size, n_categories), dtype=dtype)
        for i, label in enumerate(labels):
            # Subtract 1 from each LongTensor because your
            # indexing starts at 1 and tensor indexing starts at 0
            label = torch.LongTensor(label) - 1
            one_hot_labels[i] = one_hot_labels[i].scatter_(dim=0, index=label, value=1.)
        return one_hot_labels    


@dataclass
class DataCollatorForDota:
    
    max_length: int = 30
    pad_token_id: int = 0
    
    def __call__(self, examples):
        users = [torch.LongTensor(item[0]) for item in examples]
        users = torch.stack(users)
        items, target = self._tensorize_batch(examples)
        
        attn_mask = target.clone().detach()
        attn_mask[attn_mask != -100] = 1
        
        return users, items, target, attn_mask
            
    def _tensorize_batch(self, examples):
        max_length_batch = max(len(x[2]) for x in examples)
        max_length_batch = min([max_length_batch, self.max_length])
        
        items = [self.pad_onehot(item[1], max_length_batch) for item in examples]
        target = [torch.LongTensor(item[2][:max_length_batch]) for item in examples]

        items = torch.stack(items)
        target = pad_sequence(target, batch_first=True, padding_value=-100)

        return items, target
    
    def pad_onehot(self, example, max_length):
        N, C, D = example.shape
        if max_length > N:
            padding = torch.zeros([max_length-N, C, D], dtype=torch.float32)
            return torch.cat((example, padding), 0)
        else:
            return example[:max_length]


class MoviesDataset(data.Dataset):
    def __init__(self, config):

        """
        Args:
            config (box): hyperparameters file
        """
        # Hyperparameters
        self.data_path = config.dataset.train_data_path
        self.max_seq = config.dataset.max_seq_length
        
        # load paths of books in subfolder <data>
        logger.info("Loading features from dataset at %s", self.data_path)

        self.examples = self.open_dataset(self.data_path)
        self.user2id, self.item2id = self.load_dictionaries(self.examples)
    
    def __getitem__(self, i):
        list_users = list(self.examples[i].keys())
        champions = self.convert_to_ids(self.user2id, list_users)
        items, labels = self.convert_to_multihot(self.examples[i])
        return torch.tensor(champions, dtype=torch.long), items, torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)
    
    def open_dataset(self, path):
        pickle_in = open(path,"rb")
        data = pickle.load(pickle_in)
        pickle_in.close()
        return data
    
    def load_dictionaries(self, examples):
        user_ids = []
        item_ids = []
        for instance in examples:
            user_ids.append(list(instance.keys()))
            for key in list(instance.keys()):
                item_ids.append(list(itertools.chain(*instance[key])))
        user_ids = list(itertools.chain(*user_ids))
        item_ids = list(itertools.chain(*item_ids))
        user2id = {int(v) : i for i, v in enumerate(set(user_ids))}
        item2id = {v : i for i, v in enumerate(set(item_ids))}
        return user2id, item2id
    
    def convert_to_ids(self, map_dict, tokens):
        return [map_dict[int(i)] for i in tokens]
    
    def convert_to_multihot(self, sample):
        champs = list(sample.keys())
        actual_len = len(sample[champs[-1]])
        
        user_items = sample[champs[0]]
        user_items = list(itertools.chain(*user_items))
        labels = self.convert_to_ids(self.item2id, user_items[1:])
        labels = [-100 if x==0 else x for x in labels]
        
        list_one_hot = []
        for champ in champs:
            result = []
            item_sets = sample[champ]
            item_sets = item_sets[:actual_len]
            for i_set in item_sets:
                item_set = []
                for i in i_set:
                    item_set.append(self.item2id[int(i)]+1) # this is for shift
                result.append(item_set)
            list_one_hot.append(self.to_onehot(result, len(self.item2id), dtype=torch.float32))

        items_tensor = torch.stack(list_one_hot, dim=1) # shape S,C,D
        return items_tensor, labels

    def to_onehot(self, labels, n_categories, dtype=torch.float32):
        batch_size = len(labels)
        one_hot_labels = torch.zeros(size=(batch_size, n_categories), dtype=dtype)
        for i, label in enumerate(labels):
            # Subtract 1 from each LongTensor because your
            # indexing starts at 1 and tensor indexing starts at 0
            label = torch.LongTensor(label) - 1
            one_hot_labels[i] = one_hot_labels[i].scatter_(dim=0, index=label, value=1.)
        return one_hot_labels
    
    def extend_instance(self, raw_examples):
        examples = []
        for instance in raw_examples:
            extended_instance = []
            extended_instance.append(OrderedDict(instance))
            for i in range(1,len(instance)):
                keys = np.array(list(instance.keys()))
                extended_instance.append(
                OrderedDict(
                    (k, instance[k])
                    for k in np.roll(keys,i)
                )
                )
            examples.extend(extended_instance)
        return examples
