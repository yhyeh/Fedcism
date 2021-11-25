#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img
import os

import pdb
import time

if __name__ == '__main__':
    # reproduce randomness
    torch.manual_seed(1001)
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}/{}/'.format(
        args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.results_save)
    if not os.path.exists(os.path.join(base_dir, 'fed')):
        os.makedirs(os.path.join(base_dir, 'fed'), exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    '''
    print('type: ', type(dataset_test))
    print('len: ', len(dataset_test))
    local_data_size = []
    for idx in range(args.num_users):
        local_data_size.append(len(dict_users_train[idx]))
    print('local dataset size: ', local_data_size.sort())
    '''
    shard_path = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}/'.format(
        args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user)
    dict_save_path = os.path.join(shard_path, 'shared_dict_users.pkl')
    if os.path.exists(dict_save_path): # use old one
        print('Local data already exist!')
        with open(dict_save_path, 'rb') as handle:
            (dict_users_train, dict_users_test) = pickle.load(handle)
    else:
        print('Re dispatch data to local!')
        with open(dict_save_path, 'wb') as handle:
            pickle.dump((dict_users_train, dict_users_test), handle)
        
    # build model
    net_glob = get_model(args)
    net_glob.train()
    #print(list(net_glob.layer_hidden1.weight)[0])

    # training
    results_save_path = os.path.join(base_dir, 'fed/results.csv')

    loss_train = []
    time_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []

    for iter in range(args.epochs):
        t_geps_bgin = time.time()
        time_locals = []
        
        w_glob = None
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1) # num of selected clients
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)


        print("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))
        '''
        for idx in idxs_users:
            print('user: ', idx, '==================================')
            print('dict_users_train[idx]: ', type(dict_users_train[idx]), len(dict_users_train[idx]))
            #print(dict_users_train[idx])
        '''
        
        for idx in idxs_users: # iter over selected clients
            t_leps_bgin = time.time()

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            net_local = copy.deepcopy(net_glob)

            w_local, loss = local.train(net=net_local.to(args.device), lr=lr)
            loss_locals.append(copy.deepcopy(loss))
            # loss: a float, avg loss over local epochs over batches
            #print('loss: ', loss)


            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
            
            t_leps_end = time.time()
            time_locals.append(t_leps_end - t_leps_bgin)

        lr *= args.lr_decay # default: no decay

        # update global weights
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m) # calculate avg by dividing m

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        #loss_train.append(loss_avg)

        t_geps_end = time.time() # not include validation time
        time_glob = t_geps_end - t_geps_bgin
        time_train.append(time_glob)
        time_local_avg = sum(time_locals) / len(time_locals)

        if (iter + 1) % args.test_freq == 0:
            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}, Avg local runtime: {:.2f}, global runtime: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test, time_local_avg, time_glob))


            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = iter

            # if (iter + 1) > args.start_saving:
            #     model_save_path = os.path.join(base_dir, 'fed/model_{}.pt'.format(iter + 1))
            #     torch.save(net_glob.state_dict(), model_save_path)

            results.append(np.array([iter, loss_avg, loss_test, acc_test, best_acc, time_local_avg, time_glob]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc', 'time_local_avg', 'time_glob'])
            final_results.to_csv(results_save_path, index=False)

        if (iter + 1) % 50 == 0:
            best_save_path = os.path.join(base_dir, 'fed/best_{}.pt'.format(iter + 1))
            model_save_path = os.path.join(base_dir, 'fed/model_{}.pt'.format(iter + 1))
            torch.save(net_best.state_dict(), best_save_path)
            torch.save(net_glob.state_dict(), model_save_path)

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))