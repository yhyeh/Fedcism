import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from utils.distribution import distr_profile, cosine_similarity


def get_exp_result(dataset, distr, VI, RUN, data_distr_file, result_folder, 
                   legends, shard_per_user, frac, acc_threshold,
                   num_users=100, global_ep = 500,
                   unbalanced=True, show_fig=True, save=False):
    
    local_ep = 1
    
    fig_size = (16*0.3, 9*0.3)
    #plt.rcParams['font.size'] = 22
    
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
    #runs = os.listdir(base_dir)
    #print(runs)
    #runs = ['selection'] # diff data distribution
    #runs = ['selection1', 'selection2', 'selection3', 'selection4', 'selection5']
    #runs = ['selection1']
    #runs = ['cossim_cls_imb']
    runs = [result_folder]
    #runs = ['cossim_ar2']
    #runs = ['cossim_test']
    #print(runs)
    #outcome = os.listdir(os.path.join(base_dir, runs[0]))
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
    bacc_algos = []
    fsim_algos = []
    required_time = {'epoch':[], 'time_simu':[]}
    linemk = [':', '--', '-.', '']
    
    '''
    fig, axs = plt.subplots(len(runs)+1, sharex=True, figsize=(fig_width, 12))
    fig.suptitle('dataset size | selection cnt - shard' + str(shard_per_user))
    axs[0].plot(range(len(local_data_size)), local_data_size,label='local data size')
    '''

    for idx, run in enumerate(runs):
        print()
        print('=== {} - {} - shard{} - select{}*{} ==='.format(
              dataset, run, shard_per_user, num_users, frac))
        
        '''
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
        '''
        cossim_algos = []
        for e in [0.8]:
            for a in ['algo3_deg1']:
                fname = '{}_e{}'.format(a, e)
                if a == 'algo3_deg1':
                    for wof in ['', '_wof100']:
                        fname += wof
                cossim_algos.append(fname)


        df_result = {} # algo -> df of result
        sim_glob = {}
        #slctcnt = {}
        print('algorithms')
        algos = ['fedavg', 'oort_e0.8']+cossim_algos #+list(map(lambda s:'uni_'+s, cossim_algos))
        print(algos)
        #plt.figure(figsize=(20, 6))
        #fig, axs = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)
        fig, ax = plt.subplots(figsize=fig_size)


        for aidx, algo in enumerate(algos):
            algo_dir = os.path.join(base_dir, '{}/{}'.format(run, algo))
            if os.path.exists(algo_dir):
                results_path = os.path.join(algo_dir, "results.csv")
                #slctcnt_path = os.path.join(base_dir_utility, "selection_cnt.csv")
                #cossim_path_utility = os.path.join(base_dir_utility, "cossim_glob_uni.csv")
                #utility_path = os.path.join(base_dir_utility, "utility.csv")
                distr_glob_frac_path = os.path.join(algo_dir, 'distr_glob_frac.csv')

                df_result[algo] = pd.read_csv(results_path)
                #slctcnt[algo] = np.genfromtxt(slctcnt_path, delimiter=',')
                distr_glob_frac = np.genfromtxt(distr_glob_frac_path, delimiter=',')
                distr_zipf = np.array([1/p for p in range(1, distr_glob_frac.shape[1]+1)])
                distr_uni = np.ones(distr_glob_frac.shape[1])
                down_sample_idx = np.arange(0, round(distr_glob_frac.shape[0]), 10)

                '''
                plot global data distribution over epochs
                '''
                '''
                #print('distr_glob_frac.shape', distr_glob_frac.shape)
                plt.figure(figsize=(20, 3))
                plt.title('Global Data Distribution - {} - s{} - c{}*{} - {}'.format(run, shard_per_user, num_users, frac, algo))
                plt.xlabel('Epoch')
                plt.ylabel('Fraction of Data Volumn')
                base = np.zeros(down_sample_idx.shape[0])
                for c in range(distr_glob_frac.shape[1]):
                    c_vol = distr_glob_frac[down_sample_idx, c]
                    plt.bar(down_sample_idx, c_vol,
                            bottom=base, label='class {}'.format(c), width=8, alpha=0.3)
                    base += c_vol
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                        fancybox=True, shadow=True, ncol=5)

                '''
                sim_g_z = [cosine_similarity(distr_glob_frac[e], distr_zipf) for e in down_sample_idx]
                sim_g_u = [cosine_similarity(distr_glob_frac[e], distr_uni) for e in down_sample_idx]
                sim_glob[algo] = (sim_g_z, sim_g_u)
                #fig.ylabel('cos simlilarity')
                #fig.xlim([0, 1])
                #fig.suptitle('Global Distribution Similarity - {} - s{} - c{}*{}'.format(run, shard_per_user, num_users, frac))
                '''
                axs[0].title.set_text('sim(glob, zipf)')
                p=axs[0].plot(down_sample_idx, sim_g_z, '--', label=algo)
                axs[0].set_ylabel('cos simlilarity')
                axs[0].set_xlabel('epoch')
                axs[0].set_ylim([0.4, 1])
                
                axs[1].title.set_text('sim(glob, uni)')
                axs[1].plot(down_sample_idx, sim_g_u, '--')
                axs[1].set_xlabel('epoch')
                axs[1].set_ylim([0.4, 1])
                '''
                #ax.title.set_text('sim(glob, uni)')
                ax.plot(down_sample_idx, sim_g_u, linemk[aidx], label=legends[algo])
                ax.set_ylabel('Similarity( $\mathcal{D}_G$, $\mathcal{U}$ )')
                ax.set_xlabel('Epoch')
                #ax.set_ylim([0.4, 1])
                fsim_algos.append(sim_g_u[-1])
                
                #fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                #        fancybox=True, shadow=True, ncol=2)
                
                # Hide the right and top spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                ax.legend(fontsize=8, ncol=len(algos), frameon=False)

                if distr == 'cus': # alias
                    fig_name = 'sim_{}_vi{}_r{}'.format('sparsez', VI, RUN)
                else:            
                    fig_name = 'sim_{}_vi{}_r{}'.format(distr, VI, RUN)
                
                fig_path = os.path.join('save', dataset, 'fig', 'mid', fig_name)
                if save:
                    fig.tight_layout()
                    fig.savefig('{}.{}'.format(fig_path, 'png'), format='png', transparent=True)
                    fig.savefig('{}.{}'.format(fig_path, 'eps'), format='eps', transparent=True)
                if show_fig == False:
                    plt.close(fig)
                #cossim_glob_uni_utility = np.genfromtxt(cossim_path_utility, delimiter=',')
                #df_result[algo]['cossim_glob_uni'] = cossim_glob_uni_utility
                #utility = np.genfromtxt(utility_path, delimiter='\n')

                #print(df_result[algo].head(25))
                df_result[algo] = df_result[algo][:global_ep]
                #print(df_result[algo].shape)
                bacc = df_result[algo].loc[df_result[algo].shape[0]-1]['best_acc']

                #print(algo,', best_acc', bacc, '===================')
                #print('')
                bacc_algos.append(bacc)
            else:
                print('No {} training result.'.format(algo))


        #final_acc = top_acc = max(bacc_algos)
        final_acc = min_acc = min(bacc_algos[1:]) # exclude fedavg
        print()
        print('bacc summary: {} -> max: {}'.format(bacc_algos, final_acc))
        target_acc = final_acc * acc_threshold/100
        print('target_acc: {:.5f}, {}% of bacc'.format(target_acc, acc_threshold),'\n')
        
        if dataset == 'cifar10': acc_ceiling = 80
        elif dataset == 'cifar100': acc_ceiling = 17
        else: acc_ceiling = 100

        #['loss_avg', 'loss_test', 'acc_test', 'best_acc']
        #plt.rcParams["figure.figsize"] = (fig_width,6)
        metrics = [#('loss_avg',[-1, 2.5], 'Loss', 'loss'), 
                   #('acc_test', [0, 100]), 'Accuracy', 
                   ('best_acc', [0, acc_ceiling], 'Best Accuracy (%)', 'bacc')]
                   #('cossim_glob_uni', [0, 1], 'cossim_glob_uni')]

        for col, ylim, ylab, alias in metrics:
            for x, xlab in [('epoch', 'Epoch'), ('time_simu', 'Time')]:
                fig, ax = plt.subplots(figsize=fig_size)
                color_algos = []

                for aidx, algo in enumerate(algos):
                    algo_dir = os.path.join(base_dir, '{}/{}'.format(run, algo))

                    if os.path.exists(algo_dir):
                        #plt.plot(df_statsel[x], df_statsel[col], label='utility', marker='.')
                        p = ax.plot(df_result[algo][x], df_result[algo][col], linemk[aidx], label=legends[algo])
                        color_algos.append(p[0].get_color())

                        if col == 'best_acc':
                            try:
                                target_epc = np.where(df_result[algo]['best_acc'] >= target_acc)[0][0]
                                target_x = df_result[algo][x][target_epc]
                                required_time[x].append(target_x)
                                ax.axvline(x=target_x, ymax=target_acc/acc_ceiling,
                                            color=p[0].get_color(), ls='-', lw=1, alpha=0.5)
                            except:
                                required_time[x].append(-1)
                                target_epc = None
                                target_x = None
                                
                        '''
                        if col == 'best_acc' or 'loss_avg':
                            diff_bacc = df_result[algo][col].diff()
                            ma_diff_bacc = -diff_bacc.rolling(20).sum()

                            ma_bacc = df_result[algo][col].rolling(10).mean()
                            #diff_ma_bacc = ma_bacc.diff()
                            #print(ma_diff_bacc[20:100])
                            plt.plot(df_result[algo][x], diff_bacc, label=algo+' diff')
                            plt.plot(df_result[algo][x], ma_diff_bacc, label=algo+' MA of diff')
                            plt.plot(df_result[algo][x], ma_bacc, label=algo+' MA')
                            #plt.plot(df_result[algo][x], diff_ma_bacc, label=algo+' diff of MA')
                            print('turning point of {}: {}'.format(algo, df_result[algo][x][ma_diff_bacc.idxmin()]))
                        '''

                #plt.legend(fancybox=True, shadow=True, fontsize=16,
                #           loc='lower left', bbox_to_anchor=(1, 0))#, ncol=3)
                ax.set_ylabel(ylab)
                ax.set_xlabel(xlab)
                ax.set_ylim(ylim)
                #plt.title('{} - {} - s{} - c{}*{}'.format(dataset, run, shard_per_user, num_users, frac), y=1.1)
                #plt.title(run + ' - ' + str(shard_per_user) + ' class per client')
                if col == 'best_acc':
                    print('required', x)
                    print(required_time[x])
                    worst = max(required_time[x])
                    print(['{:.2f}x'.format(worst/t) for t in required_time[x]])
                    ax.axhline(y=target_acc, color='grey', ls='-', lw=1, alpha=0.5)
                    ax.text(0, target_acc+1, 'Target', ha='left', color='grey', fontsize=8, fontfamily='monospace')

                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                ax.legend(fontsize=8, ncol=len(algos), frameon=False)

                if distr == 'cus': # alias
                    fig_name = '{}-{}_{}_s{}_c{}_vi{}_r{}'.format(alias, xlab, 'sparsez', shard_per_user, frac, VI, RUN)
                else:            
                    fig_name = '{}-{}_{}_s{}_c{}_vi{}_r{}'.format(alias, xlab, distr, shard_per_user, frac, VI, RUN)
                
                fig_path = os.path.join('save', dataset, 'fig', 'mid', fig_name)
                fig.tight_layout()

                if save:
                    fig.savefig('{}.{}'.format(fig_path, 'png'), format='png', transparent=True)
                    fig.savefig('{}.{}'.format(fig_path, 'eps'), format='eps', transparent=True)
                
                if not show_fig: 
                    plt.close()
                    #fig.close()

        '''
        plot selection cnt
        '''
        '''
        set_labels = [(x, y) for x in algos for y in ['exploitation','exploration']]
        for x in ['epoch']: #['epoch', 'time_simu']:
            plt.figure(figsize=(20, 15))
            for i, algo in enumerate(algos):
                #print(df_result[algo].head())
                for slct_type, mk, lb in [('user_explor', 'x', 'exploration'),('user_exploi', 'o', 'exploitation')]:
                    if algo == 'fedavg': continue #and slct_type == 'user_exploi': continue
                    candidates = df_result[algo][slct_type].tolist() # [str, str,...]
                    #print(type(candidates[0]))

                    candidates = list(map(json.loads, candidates)) # dim: num epochs x num candidates
                    #print(type(candidates.iloc[0]), type(candidates.iloc[1]))
                    #print(type(candidates[0]))

                    if (algo, lb) in set_labels:
                        for epc in range(len(df_result[algo][x])):
                            if epc == 0:
                                plt.scatter([df_result[algo][x][epc]]*len(candidates[epc]), np.array(candidates[epc])+i*0.5, 
                                            c=color_algos[i], alpha=0.5, marker=mk, facecolors='none', label='{} {}'.format(algo, lb))
                                set_labels.remove((algo, lb))
                            else:
                                plt.scatter([df_result[algo][x][epc]]*len(candidates[epc]), np.array(candidates[epc])+i*0.5, 
                                            c=color_algos[i], alpha=0.5, marker=mk, facecolors='none')
                                
                    else:
                        for epc in range(len(df_result[algo][x])):
                            plt.scatter([df_result[algo][x][epc]]*len(candidates[epc]), np.array(candidates[epc])+i*0.5, 
                                        c=color_algos[i], alpha=0.5, marker=mk, facecolors='none')
                                
            plt.grid(which='major', color='#DDDDDD', linewidth=0.6)
            plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.4)
            plt.minorticks_on()
            plt.ylabel('client ID')
            plt.xlabel(x)
            plt.legend(fancybox=True, shadow=True, fontsize=20, ncol=2,
                       loc='upper center', bbox_to_anchor=(0.5, -0.2))
            plt.title('Selection Record\n{} - {} - s{} - c{}*{}'.format(dataset, run, shard_per_user, num_users, frac))
        '''
        '''# old one
        plt.figure(figsize=(20, 6))
        for i, algo in enumerate(utility_algos):
            plt.bar(np.arange(len(slctcnt[algo]))+i*0.2, slctcnt[algo],
                    width=0.2, label=algo, color = color_algos[i+1])
        plt.ylabel('selection cnt')
        plt.xlabel('client id')
        plt.legend(fancybox=True, shadow=True, fontsize=20, ncol=4,
                   loc='upper center', bbox_to_anchor=(0.5, -0.2))
        '''
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

    return algos, color_algos, bacc_algos, required_time, fsim_algos