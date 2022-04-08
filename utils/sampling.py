#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import random
from itertools import permutations
import numpy as np
import torch
import pdb
from scipy.stats import norm

def fair_iid(dataset, num_users):
    """
    Sample I.I.D. client data from fairness dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def fair_noniid(train_data, num_users, num_shards=200, num_imgs=300, train=True, rand_set_all=[]):
    """
    Sample non-I.I.D client data from fairness dataset
    :param dataset:
    :param num_users:
    :return:
    """
    assert num_shards % num_users == 0
    shard_per_user = int(num_shards / num_users)

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)

    #import pdb; pdb.set_trace()

    labels = train_data[1].numpy().reshape(len(train_data[0]),)
    assert num_shards * num_imgs == len(labels)
    #import pdb; pdb.set_trace()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    if len(rand_set_all) == 0:
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, shard_per_user, replace=False))
            for rand in rand_set:
                rand_set_all.append(rand)

            idx_shard = list(set(idx_shard) - rand_set) # remove shards from possible choices for other users
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    else: # this only works if the train and test set have the same distribution of labels
        for i in range(num_users):
            rand_set = rand_set_all[i*shard_per_user: (i+1)*shard_per_user]
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    return dict_users, rand_set_all

def iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def noniid_unbalanced(dataset, num_users, shard_per_user, rand_set_all=[],
                      cls_imbalance=False, clsimb_type='zipf'):
    """
    On top of noniid with customized rand_set_all / unbalanced_rand_set
    :param dataset:
    :param num_users:
    :return:
    """

    if len(rand_set_all) != 0: # just a wrap of non_iid() for test_data
        return noniid(dataset, num_users, shard_per_user, rand_set_all)
    
    else:
        num_classes = len(np.unique(dataset.targets))
        num_samples = len(dataset)
        num_shards = shard_per_user * num_users
        
        if cls_imbalance:
            balance_shard_per_class = int(shard_per_user * num_users / num_classes)
            shard_per_class = np.array([balance_shard_per_class]*num_classes)
            # dtype of shard_per_class becomes list instead of const

            if clsimb_type == 'zipf':
                zipf_ratio = 1/np.arange(1, num_classes+1)
                raw_shard_per_class = shard_per_class * zipf_ratio # down sampled
                shard_per_class = raw_shard_per_class / raw_shard_per_class.sum() * num_shards
            elif clsimb_type == 'htail':
                # make class 0 keep original volume, other reduced to 10%
                shard_per_class *= 0.1
                shard_per_class[0] *= 10
            #elif clsimb_type == 'gaussian':
            else:
                # make class 0 as rare as 10% of original volume
                shard_per_class[0] *= 0.1
            
            shard_per_class = np.ceil(shard_per_class).astype(int)
            all_shards = []
            for c in range(num_classes):
                all_shards.extend([c]*shard_per_class[c])
            #print('shard_per_class', shard_per_class)
            #print('all_shards', all_shards)
        else:
            shard_per_class = int(shard_per_user * num_users / num_classes)
            all_shards = list(range(num_classes)) * shard_per_class

        np.random.shuffle(all_shards)
        
        # each user get unbalanced num of shards, the num followed normal distribution
        pdf = norm.pdf(range(num_users) , loc = int(num_users/2) , scale = int(num_users/3))
        pdf = pdf / pdf.sum()
        # guarantee there are at least 1 shard per user
        assert len(all_shards) > num_users
        shard_owner = np.random.choice(range(num_users), size=len(all_shards)-num_users, p = pdf)
        shard_owner = np.concatenate((shard_owner, range(num_users)), axis=None)

        assert(len(all_shards) == len(shard_owner))
        #rand_set_all = np.array(rand_set_all).reshape((num_users, -1))
        
        unbalanced_rand_set = []
        for i in range(num_users): unbalanced_rand_set.append([])
        for shard, owner in zip(all_shards, shard_owner):
            unbalanced_rand_set[owner].append(shard)

        return noniid(dataset, num_users, shard_per_user, unbalanced_rand_set)

def noniid(dataset, num_users, shard_per_user, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # userID : [local datapointIDs]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # label : [datapointIDs]
    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    # reshape (group) data as (num_shard, data per shard) for each class
    num_classes = len(np.unique(dataset.targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x # dim: (shard, data point), type: list<np.array>

    # rand_set_all: user -> [shard for class1, shard for class3, ...]
    if len(rand_set_all) == 0: # without customized rand_set_all
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # statistical of data label distribution
    distr_users = {} # userID : [123, (# data in class), ..., 320]->len=num_classes
    
    # divide and assign
    idxs_dict_len = {k:len(v) for k, v in idxs_dict.items()}
    idxs_dict_rec = {k:list(range(len(v))) for k, v in idxs_dict.items()}
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        distr_users[i] = [0 for i in range(num_classes)]
        for label in rand_set_label:
            #print('class {} remaining shard opts {}/{}'.format(label, len(idxs_dict_rec[label]), len(idxs_dict[label])))

            if len(idxs_dict_rec[label]) == 0: # all visited, renew dict (over sampling)
                idxs_dict_rec[label] = list(range(len(idxs_dict[label])))
                #print('class {} dict renewed:{}'.format(label, idxs_dict_rec[label]))
            
            shard_idx = np.random.choice(idxs_dict_rec[label], replace=False) # randomly pick a shard
            idxs_dict_rec[label].remove(shard_idx)
            distr_users[i][label] += len(idxs_dict[label][shard_idx])
            rand_set.append(idxs_dict[label][shard_idx])
            
            

        dict_users[i] = np.concatenate(rand_set)

    # test to guarantee user get data with noniidness
    '''
    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        assert(len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert(len(test) == len(dataset))
    assert(len(set(list(test))) == len(dataset))
    '''
    # test of distr_users
    for u in range(num_users):
        assert len(dict_users[u]) == sum(distr_users[u])
    
    return dict_users, rand_set_all, distr_users

def noniid_replace(dataset, num_users, shard_per_user, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    imgs_per_shard = int(len(dataset) / (num_users * shard_per_user))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    if len(rand_set_all) == 0:
        for i in range(num_users):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users[i] = np.concatenate(rand_set)

    for key, value in dict_users.items():
        assert(len(np.unique(torch.tensor(dataset.targets)[value]))) == shard_per_user

    return dict_users, rand_set_all