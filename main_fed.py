#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch
from torchinfo import summary

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
    np.random.seed(1001)
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
    dict_save_path = os.path.join(shard_path, 'unbalanced_dict_users.pkl')
    if os.path.exists(dict_save_path): # use old one
        print('Local data already exist!')
        with open(dict_save_path, 'rb') as handle:
            (dict_users_train, dict_users_test) = pickle.load(handle)
    else:
        print('Re dispatch data to local!')
        with open(dict_save_path, 'wb') as handle:
            pickle.dump((dict_users_train, dict_users_test), handle)
        
    # build cloud model
    net_glob = get_model(args)
    
    # get model size
    glob_summary = summary(net_glob)
    print(glob_summary)
    net_size = glob_summary.total_params

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

    ### simulate dynamic training + tx time
    time_simu = 0
    time_save_path= './save/user_config/var_time/{}_{}.csv'.format(args.dataset, args.num_users)
    if os.path.exists(time_save_path):
        # load shared config
        print('Load existed time config...')
        t_all = np.genfromtxt(time_save_path, delimiter=',')
    else:
        # generate new config and save
        print('Generate new time config...')
        t_all = np.zeros((args.num_users, args.epochs))
        t_mean = np.random.randint(1, 5, args.num_users) # rand choose from 1~10
        for u in range(args.num_users):
            t_all[u] = np.random.poisson(t_mean[u], size=args.epochs) + 1

    for iter in range(args.epochs):
        t_geps_bgin = time.time()
        #time_locals = []
        t_local = t_all[:, iter]
        
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
                for k in w_glob.keys(): # layer by layer
                    w_glob[k] += w_local[k]
            
            t_leps_end = time.time()
            #time_locals.append(t_leps_end - t_leps_bgin)
            #time_locals.append(t_local[idx])

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
        #time_local_avg = sum(time_locals) / len(time_locals)
        time_local_max = max(t_local[idxs_users])
        time_simu += time_local_max

        if (iter + 1) % args.test_freq == 0:
            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}, Max local runtime: {:.2f}, Simu runtime: {:.2f}, global runtime: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test, time_local_max, time_simu, time_glob))


            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = iter

            # if (iter + 1) > args.start_saving:
            #     model_save_path = os.path.join(base_dir, 'fed/model_{}.pt'.format(iter + 1))
            #     torch.save(net_glob.state_dict(), model_save_path)

            results.append(np.array([iter, loss_avg, loss_test, acc_test, best_acc, time_local_max, time_simu, time_glob]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc', 'time_local_max', 'time_simu', 'time_glob'])
            final_results.to_csv(results_save_path, index=False)
        '''
        if (iter + 1) % 50 == 0:
            best_save_path = os.path.join(base_dir, 'fed/best_{}.pt'.format(iter + 1))
            model_save_path = os.path.join(base_dir, 'fed/model_{}.pt'.format(iter + 1))
            torch.save(net_best.state_dict(), best_save_path)
            torch.save(net_glob.state_dict(), model_save_path)
        '''
    np.savetxt(time_save_path, t_all, delimiter=",")
    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))