#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch
import scipy.stats as st
from torchinfo import summary
from math import ceil

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from utils.distribution import cosine_similarity, distr_profile
from models.Update import LocalUpdate
from models.test import test_img
import os

import pdb
import time
import json
import datetime
import termplotlib as tpl

if __name__ == '__main__':
    t_prog_bgin = time.time()
    # reproduce randomness
    torch.manual_seed(1001)
    np.random.seed(1001)

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    algo_dir = 'algo{}_r{}'.format(args.myalgo, args.gamma)
    base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}/{}/'.format(
        args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.results_save)
    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test, distr_users, _ = get_data(args)
    # dict_users_test is unused actually
    
    #print('type: ', type(dataset_test))
    #print('len: ', len(dataset_test))
    #print(distr_users)
    distr_uni = np.ones(args.num_classes)
    distr_uni = distr_uni / np.linalg.norm(distr_uni)

    shard_path = './save/{}/data_distr/num{}/shard{}/'.format(
        args.dataset, args.num_users, args.shard_per_user)
    dict_save_path = os.path.join(shard_path, args.data_distr)
    if os.path.exists(dict_save_path): # use old one
        print('Local data already exist!')
        with open(dict_save_path, 'rb') as handle:
            (dict_users_train, dict_users_test, distr_users) = pickle.load(handle)
    else:
        print('Re dispatch data to local!')
        with open(dict_save_path, 'wb') as handle:
            pickle.dump((dict_users_train, dict_users_test, distr_users), handle)
            os.chmod(dict_save_path, 0o444) # read-only
        
    '''
    local_data_size = []
    for idx in range(args.num_users):
        local_data_size.append(len(dict_users_train[idx]))
    print('local dataset size: ', local_data_size)
    '''

    # build model
    net_glob = get_model(args)

    ##################
    # get model size #
    ##################
    glob_summary = summary(net_glob)
    #print(glob_summary)
    net_size = glob_summary.total_params


    ############
    # training #
    ############
    net_glob.train()
    
    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')
    slctcnt_save_path = os.path.join(base_dir, algo_dir, 'selection_cnt.csv')
    utility_save_path = os.path.join(base_dir, algo_dir, 'utility.csv')
    if os.path.exists(utility_save_path): # delete
        os.remove(utility_save_path)

    loss_train = []
    time_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []

    epsilon = 0.5 # exploitation rate
    alpha = 0.1 # penalty exp factor
    confd = 0.95 
    slct_cnt = np.zeros(args.num_users)
    utility_hist = {} # clientID : utility
    utility_stat = {} # clientID : utility
    T = 5
    cossim_glob_uni = np.zeros(args.epochs)
    cossim_glob_uni_path = os.path.join(base_dir, algo_dir, 'cossim_glob_uni.csv')
    bacc_wndw_size = args.wndw_size
    bacc_wndw = np.ones(bacc_wndw_size)
    best_acc_prev = None
    BACC_STABLE = False
    stable_epoch = None

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

        if iter == 0: # first round
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            distr_glob = np.zeros(args.num_classes) # normalized
        else:
            '''
            keep top good_cli = m*epsilon utility client
            drop & random select (1-good_cli)
            '''
            n_exploi = round(m*epsilon)
            n_explor = m - round(m*epsilon)
            sorted_u = {}
            # ascending [explor...exploi]
            for i, (k, v) in enumerate(sorted(utility_hist.items(), key=lambda item: item[1])):
                sorted_u[k] = v
                if i == len(utility_hist)-n_exploi:
                    cutoff_utility = v * confd

            # descending
            #sorted_u = {k: v for k, v in sorted(utility_hist.items(), key=lambda item: item[1], reverse=True)}

            # err: misunderstood of confd
            #lob, upb = st.t.interval(confd, len(sorted_u.values())-1, loc=np.mean(sorted_u.values()), scale=st.sem(sorted_u.values()))            
            
            ### 
            n_clip = ceil(len(sorted_u)*(1-confd))
            n_pool = len(sorted_u) # init
            pool_util = {} # candidate of this round
            for k, v in sorted_u.items():
                if v < cutoff_utility:
                    n_pool -= 1
                    continue
                else:
                    if len(pool_util) <= n_pool - n_clip: # util // prob
                        pool_util[k] = v
                        if len(pool_util) == n_pool - n_clip: # util upper bound
                            util_upb = v
                    else: # clipped part
                        pool_util[k] = util_upb
            
            #print('=== utility ===', len(sorted_u),'\n', sorted_u)
            #print('=== cutoff_utility ===\n', cutoff_utility)
            #print('=== pool_util ===', len(pool_util),'\n', pool_util)
    
            ### sample n_exploi clients by util from pool
            pdf = np.array(list(pool_util.values()))/sum(pool_util.values())
            user_exploi = np.random.choice(list(pool_util.keys()), n_exploi, p=pdf, replace=False)
            #print('exploi users', user_exploi)
            assert len(user_exploi) == n_exploi

            ### sample at most n_explor clients by speed from unexplored
            rest_pool = list(set(range(args.num_users)) - set(pool_util.keys()))
            #print('=== rest_pool ===', len(rest_pool),'\n', rest_pool)
            if len(rest_pool) > n_explor: # require sampling
                rest_sel = np.random.choice(rest_pool, n_explor, replace=False)# random sample
                #pdf = t_local[rest_pool]/sum(t_local[rest_pool])
                #rest_sel = np.random.choice(rest_pool, n_explor, p=pdf, replace=False) # sample by speed
                #rest_sel = sorted(t_local[rest_pool], reverse=True)[:n_explor+1] # top-n_explor by speed
                idxs_users = np.concatenate((user_exploi, rest_sel))
            
            elif len(rest_pool) > 0: # all included, final num of participants will be less than m
                idxs_users = np.concatenate((user_exploi, rest_pool))
            else:
                idxs_users = user_exploi

            print('slct users', idxs_users)
            assert len(idxs_users) <= m

        print("\n Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))
        '''
        for idx in idxs_users:
            print('user: ', idx, '==================================')
            print('dict_users_train[idx]: ', type(dict_users_train[idx]), len(dict_users_train[idx]))
            #print(dict_users_train[idx])
        '''

        #utility_slct = {} # clientID : utility 

        
        # In ADVANCE calculate global data distribution 
        for idx in idxs_users:
            distr_glob += distr_users[idx]
        #distr_glob = distr_glob / np.linalg.norm(distr_glob)
        distr_glob = distr_glob / sum(distr_glob) # indicate the portion of label
        print('global distribution after round {}(%): {}'.format(iter, [format(100*x, '3.2f') for x in distr_glob]))
        '''
        fig = tpl.figure()
        fig.barh([round(100*x) for x in distr_glob], range(10), force_ascii=True)
        fig.show()
        '''

        for idx in idxs_users: # iter over selected clients
            t_leps_bgin = time.time()

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            net_local = copy.deepcopy(net_glob)

            w_local, loss = local.train(net=net_local.to(args.device), lr=lr)
            loss_locals.append(copy.deepcopy(loss))
            # loss: a float, avg loss over local epochs over batches
            #print('loss: ', loss)

            
            # calculate utility for next round
            B_i = len(dict_users_train[idx])
            #if int(idx) in utility_stat.keys():
            #    utility_hist[int(idx)] = utility_stat[int(idx)]
            #else:
            
            if args.myalgo == 0: # pure utility
                cossim_factor = 1
            
            elif args.myalgo == 2: # adaptive
                if iter == 0:
                    cossim_factor = 1
                else:
                    # new gamma depends on 1/loss_avg
                    print('client {} distribution(%): {}'.format(idx, [format(100*x/B_i, '3.2f') for x in distr_users[idx]]))

                    cossim_factor = 1+(args.gamma/loss_avg)**2*(1-cosine_similarity(distr_users[idx], distr_glob))
                    print('cossim_factor = {:.10f} = 1+{:.3f}*{:.5f}'.format(cossim_factor,
                                                                args.gamma/loss_avg,
                                                                1-cosine_similarity(distr_users[idx], distr_glob)))
            else: # const
                # find complementary client
                cossim_factor = 1+args.gamma*(1-cosine_similarity(distr_users[idx], distr_glob))
                #print('cossim utility, cossim_factor: ', cossim_factor)
            '''
            elif args.myalgo == 2 and not BACC_STABLE:
                # find similar client
                #cossim_factor = 1+args.gamma*(cosine_similarity(distr_users[idx], distr_glob))
                cossim_factor = 1
                print('original utility, bacc_wndw.sum():', bacc_wndw.sum())
            '''
            utility_hist[int(idx)] = utility_stat[int(idx)] = np.sqrt(B_i*loss**2)*cossim_factor

            
            # consider system hetero
            if T < t_local[idx]:
                utility_hist[int(idx)] *= (T/t_local[idx]) ** alpha

            slct_cnt[idx] += 1

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
            
            t_leps_end = time.time()
            #time_locals.append(t_leps_end - t_leps_bgin + t_local[idx])
            #time_locals.append(t_local[idx])


        lr *= args.lr_decay # default: no decay

        # update global weights
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m) # calculate avg by dividing m

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        cossim_glob_uni[iter] = cosine_similarity(distr_glob, distr_uni)

        #print('global distribution: ', distr_glob)
        print('cossim(global, uniform): ', cossim_glob_uni[iter])

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

            best_acc_prev = best_acc
            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = iter

            if best_acc_prev is not None and not BACC_STABLE:
                bacc_wndw[iter % bacc_wndw_size] = best_acc - best_acc_prev

                if iter > bacc_wndw_size and bacc_wndw.sum() == 0:
                    BACC_STABLE = True
                    stable_epoch = iter
            # if (iter + 1) > args.start_saving:
            #     model_save_path = os.path.join(base_dir, algo_dir, 'model_{}.pt'.format(iter + 1))
            #     torch.save(net_glob.state_dict(), model_save_path)

            #print('=== mov_sum ===', bacc_wndw.sum())
            #print(bacc_wndw)
            #print('Best acc stable point: epoch ', stable_epoch)

            results.append(np.array([iter, loss_avg, loss_test, acc_test, best_acc, time_local_max, time_simu, time_glob, bacc_wndw.sum()]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc', 'time_local_max', 'time_simu', 'time_glob', 'mov_sum'])
            final_results.to_csv(results_save_path, index=False)
            np.savetxt(slctcnt_save_path, slct_cnt, delimiter=",")
        ''' 
        if (iter + 1) % 50 == 0:
            best_save_path = os.path.join(base_dir, algo_dir, 'best_{}.pt'.format(iter + 1))
            model_save_path = os.path.join(base_dir, algo_dir, 'model_{}.pt'.format(iter + 1))
            torch.save(net_best.state_dict(), best_save_path)
            torch.save(net_glob.state_dict(), model_save_path)
        '''
    # save history of utility of all explored clients
    with open(utility_save_path, 'w') as fp:
                    fp.write("{}\n".format(json.dumps(sorted_u)))
    np.savetxt(time_save_path, t_all, delimiter=",")
    np.savetxt(cossim_glob_uni_path, cossim_glob_uni, delimiter=",")

    t_prog = time.time() - t_prog_bgin
    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))
    print('Program execution time:', datetime.timedelta(seconds=t_prog))
    