import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


def get_exp_result(dataset, data_distr_file, result_folder, 
                   shard_per_user, frac, acc_threshold, unbalanced=True):
    
    num_users = 100
    local_ep = 1
    global_ep = 1000
    
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams['font.size'] = 22
    
    if dataset == 'mnist':
        model = 'mlp'
    else:
        model = 'cnn'

    iid = False
    if unbalanced:
        #data_dict_fname = 'unbalanced_dict_users_2.pkl'
        data_dict_fname = data_distr_file
    else:
        data_dict_fname = 'shared_dict_users.pkl'

        if shard_per_user == 10:
            iid = True

    #save\mnist\mlp_iidTrue_num100_C0.1_le1\shard10\run1\fed
    base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}/'.format(
        dataset, model, iid, num_users, frac, local_ep, shard_per_user)
    runs = os.listdir(base_dir)
    #print(runs)
    #runs = ['selection'] # diff data distribution
    #runs = ['selection1', 'selection2', 'selection3', 'selection4', 'selection5']
    #runs = ['selection1']
    #runs = ['cossim_cls_imb']
    runs = [result_folder]
    #runs = ['cossim_ar2']
    #runs = ['cossim_test']
    #print(runs)
    outcome = os.listdir(os.path.join(base_dir, runs[0]))
    #print(outcome)

    shard_path = './save/{}/data_distr/num{}/shard{}/'.format(
            dataset, num_users, shard_per_user)
    data_dict_path = os.path.join(shard_path, data_dict_fname)
    with open(data_dict_path, 'rb') as handle:
        (dict_users_train, dict_users_test, _) = pickle.load(handle)
    local_data_size = []
    for idx in range(num_users):
        local_data_size.append(len(dict_users_train[idx]))
    #print('local dataset size: ', local_data_size)


    #frac = 0.1
    acc_fed = np.zeros(len(runs))
    acc_utility = np.zeros(len(runs))
    acc_local_localtest = np.zeros(len(runs))
    acc_local_newtest_avg = np.zeros(len(runs))
    acc_local_newtest_ens = np.zeros(len(runs))
    lg_metrics = {}
    bacc_algos = []
    required_time = {'epoch':[], 'time_simu':[]}
    acc_algo = []
    
    '''
    fig, axs = plt.subplots(len(runs)+1, sharex=True, figsize=(fig_width, 12))
    fig.suptitle('dataset size | selection cnt - shard' + str(shard_per_user))
    axs[0].plot(range(len(local_data_size)), local_data_size,label='local data size')
    '''

    for idx, run in enumerate(runs):
        print()
        print('=== {} - {} - shard{} - select{}*{} ==='.format(
              dataset, run, shard_per_user, num_users, frac))
        # FedAvg

        base_dir_fed = os.path.join(base_dir, "{}/fedavg_c{}".format(run, frac))
        if os.path.exists(base_dir_fed):
            results_path_fed = os.path.join(base_dir_fed, "results.csv")
            cossim_path_fed = os.path.join(base_dir_fed, "cossim_glob_uni.csv")
            cossim_glob_uni_fed = np.genfromtxt(cossim_path_fed, delimiter=',')

            df_fed = pd.read_csv(results_path_fed)
            df_fed['cossim_glob_uni'] = cossim_glob_uni_fed
            #print(df_fed.head(20))
            df_fed = df_fed[:global_ep]
            #print(df_fed.shape)
            acc_fed[idx] = df_fed.loc[df_fed.shape[0]-1]['best_acc']
            acc_algo.append(acc_fed[idx])
            #print('fedavg, best_acc', acc_fed[idx], '===================')
            #print('')
            bacc_algos.append(acc_fed[idx])
        else:
            print('No fedavg training result.')

        # fedavg w/ utility selection
        #gamma_opt = [0.1, 0.2, 0.4, 0.8, 1.6]
        #gamma_opt = [1, 2, 4, 8]
        #gamma_opt = [0.1, 0.2, 0.4, 0.8, 1, 1.6, 2, 4, 8]
        #gamma_path = ['gamma_3090/{}/algo1'.format(r) for r in gamma_opt]
        #utility_algos = ['utility_cossim']
        #utility_algos = ['utility_cossim', 'algo1', 'algo2','algo2_2']
        #utility_algos = ['utility_cossim', 'var_utility', 'algo1_r1.6','algo2_r1.6']
        #utility_algos = ['utility_c0.1']

        if shard_per_user == 10:
            utility_algos = ['utility_c0.1', '_algo1_r1.6', 'algo2_r-loss']
            #utility_algos = ['utility_c0.1', 'algo1_r1.6']
        elif shard_per_user == 2:
            utility_algos = ['utility_c0.1', '_algo1_r0.8', 'algo2_r-loss']
            #utility_algos = ['utility_c0.1', 'algo1_r0.8']
        #utility_algos.extend(gamma_path)

        #utility_algos = ['utility_c0.1', 'algo1_r0.1', 'algo1_r1.0', 'algo1_r2.0']
        utility_algos = [#'utility', 

                         #'algo1_r1.0',#'algo2_sq', 
                         #'algo2_loss-ratio',
                         #'algo3_deg1', 'algo3_deg3', 'algo3_deg5'
                         #'algo3_deg1_e0.8', 'algo3_deg3_e0.8', 'algo3_deg5_e0.8'
                         #'algo5_deg5'
                         #'algo5_deg1', 'algo5_deg3', 'algo5_deg5', 'algo5_deg7'#, 'algo5_deg9'
                        ]
        for e in [0.8]:
            for a in ['oort', 'algo3_deg1']:
                fname = '{}_e{}'.format(a, e)
                if a == 'algo3_deg1':
                    for wof in ['', '_wof100']:
                        fname += wof
                utility_algos.append(fname)


        df_utility = {} # algo -> df of result
        slctcnt = {}
        print('algorithms')
        algos = ['fedavg']+utility_algos
        print(algos)
        for algo in utility_algos:
            base_dir_utility = os.path.join(base_dir, '{}/{}'.format(run, algo))
            if os.path.exists(base_dir_utility):
                results_path_utility = os.path.join(base_dir_utility, "results.csv")
                slctcnt_path = os.path.join(base_dir_utility, "selection_cnt.csv")
                #cossim_path_utility = os.path.join(base_dir_utility, "cossim_glob_uni.csv")
                utility_path = os.path.join(base_dir_utility, "utility.csv")
                df_utility[algo] = pd.read_csv(results_path_utility)
                slctcnt[algo] = np.genfromtxt(slctcnt_path, delimiter=',')
                #cossim_glob_uni_utility = np.genfromtxt(cossim_path_utility, delimiter=',')
                #df_utility[algo]['cossim_glob_uni'] = cossim_glob_uni_utility
                #utility = np.genfromtxt(utility_path, delimiter='\n')

                #print(df_utility[algo].head(25))
                df_utility[algo] = df_utility[algo][:global_ep]
                #print(df_utility[algo].shape)
                acc_utility[idx] = df_utility[algo].loc[df_utility[algo].shape[0]-1]['best_acc']
                acc_algo.append(acc_utility[idx])

                #print(algo,', best_acc', acc_utility[idx], '===================')
                #print('')
                bacc_algos.append(acc_utility[idx])
            else:
                print('No {} training result.'.format(algo))


        top_acc = max(bacc_algos)
        print()
        print('bacc summary: {} -> max: {}'.format(bacc_algos, top_acc))
        target_acc = top_acc * acc_threshold/100
        print('target_acc: {:.5f}, {}% of bacc'.format(target_acc, acc_threshold),'\n')
        if dataset == 'cifar10':
            acc_ceiling = 70
        else:
            acc_ceiling = 17

        #['loss_avg', 'loss_test', 'acc_test', 'best_acc']
        #plt.rcParams["figure.figsize"] = (fig_width,6)
        metrics = [#('loss_avg',[-1, 2.5]), 
                   #('acc_test', [0, 100]), 
                   ('best_acc', [0, acc_ceiling])]
                   #('cossim_glob_uni', [0, 1])]

        for col, yl in metrics:
            for x in ['epoch', 'time_simu']:
                plt.figure()
                color_algos = []

                if os.path.exists(base_dir_fed):
                    #plt.plot(df_fed[x], df_fed[col], label='random', marker='^')
                    p = plt.plot(df_fed[x], df_fed[col], label='random')
                    color_algos.append(p[0].get_color())
                    if col == 'best_acc':
                        try:
                            target_epc = np.where(df_fed['best_acc'] >= target_acc)[0][0]
                            target_x = df_fed[x][target_epc]
                            required_time[x].append(target_x)
                            plt.axvline(x=target_x, ymax=target_acc/acc_ceiling,
                                        color=p[0].get_color(), ls='--', lw=1)
                        except:
                            required_time[x].append(-1)
                            target_epc = None
                            target_x = None

                for algo in utility_algos:
                    base_dir_utility = os.path.join(base_dir, '{}/{}'.format(run, algo))

                    if os.path.exists(base_dir_utility):
                        #plt.plot(df_statsel[x], df_statsel[col], label='utility', marker='.')
                        p = plt.plot(df_utility[algo][x], df_utility[algo][col], label=algo)
                        color_algos.append(p[0].get_color())

                        if col == 'acc_test' or col == 'best_acc':
                            bacc = df_utility[algo]['best_acc'].iloc[-1]
                            bepc = df_utility[algo]['best_acc'].idxmax()
                            px = df_utility[algo][x][bepc]
                            max_px = df_utility[algo][x].iloc[-1]
                            #plt.plot([px], [bacc], 'o')
                            #plt.axvline(x=px, color='grey')
                            if col == 'best_acc':
                                target_epc = np.where(df_utility[algo]['best_acc'] >= target_acc)[0][0]
                                target_x = df_utility[algo][x][target_epc]
                                required_time[x].append(target_x)
                                plt.axvline(x=target_x, ymax=target_acc/acc_ceiling, 
                                            color=p[0].get_color(), ls='--', lw=1)
                                #print(p[0].get_color())
                        '''
                        if col == 'best_acc' or 'loss_avg':
                            diff_bacc = df_utility[algo][col].diff()
                            ma_diff_bacc = -diff_bacc.rolling(20).sum()

                            ma_bacc = df_utility[algo][col].rolling(10).mean()
                            #diff_ma_bacc = ma_bacc.diff()
                            #print(ma_diff_bacc[20:100])
                            plt.plot(df_utility[algo][x], diff_bacc, label=algo+' diff')
                            plt.plot(df_utility[algo][x], ma_diff_bacc, label=algo+' MA of diff')
                            plt.plot(df_utility[algo][x], ma_bacc, label=algo+' MA')
                            #plt.plot(df_utility[algo][x], diff_ma_bacc, label=algo+' diff of MA')
                            print('turning point of {}: {}'.format(algo, df_utility[algo][x][ma_diff_bacc.idxmin()]))
                        '''

                plt.legend(fancybox=True, shadow=True, fontsize=16,
                           loc='lower left', bbox_to_anchor=(1, 0))#, ncol=3)
                plt.ylabel(col)
                plt.xlabel(x)
                plt.ylim(yl)
                plt.title('{} - {} - s{} - c{}*{}'.format(dataset, run, shard_per_user, num_users, frac))
                #plt.title(run + ' - ' + str(shard_per_user) + ' class per client')
                if col == 'best_acc':
                    print('required', x)
                    print(required_time[x])
                    worst = max(required_time[x])
                    print(['{:.2f}x'.format(worst/t) for t in required_time[x]])
                    plt.axhline(y=target_acc, color='grey', ls='--', lw=1)


        plt.figure(figsize=(20, 6))
        for i, algo in enumerate(utility_algos):
            plt.bar(np.arange(len(slctcnt[algo]))+i*0.2, slctcnt[algo],
                    width=0.2, label=algo, color = color_algos[i+1])
        plt.ylabel('selection cnt')
        plt.xlabel('client id')
        plt.legend(fancybox=True, shadow=True, fontsize=20, ncol=4,
                   loc='upper center', bbox_to_anchor=(0.5, -0.2))

        #axs[idx+1].bar(range(len(slctcnt['utility_cossim'])), slctcnt['utility_cossim'])
        '''
        with open(utility_path) as fp_utility:
            plot_selection(fp_utility)
        '''


    # final acc plot for multiple runs
    '''    
    plt.rcParams["figure.figsize"] = (20,6)
    #plt.rcParams['font.size'] = 22
    plt.figure()
    plt.title('final acc')
    plt.plot(range(len(acc_fed)), acc_fed, label='random')    
    plt.plot(range(len(acc_utility)), acc_utility, label=algo + ' utility')
    plt.xlabel('run')
    plt.ylabel('acc')
    plt.legend()
    '''
    
    return algos, color_algos, bacc_algos, required_time