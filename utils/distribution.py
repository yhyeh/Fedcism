import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

def cosine_similarity(x : np.ndarray, y : np.ndarray) -> float:
    res = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return res

class distr_profile:
    '''
    A class to get info given a data distribution file
    '''

    def __init__(self, config_path):
        if not os.path.exists(config_path):
            print('Distribution config file does not exist!')
            return
        with open(config_path, 'rb') as handle:
            (self.dict_users_train, _, self.dict_distr) = pickle.load(handle)
            self.global_distr = None
            self.num_users = len(self.dict_distr)
            self.num_classes = len(self.dict_distr[0])
            self.filename = os.path.split(config_path)[1]

            # dim: num_users
            self.local_vol = np.zeros(self.num_users, dtype=int)

            # dim: num_classes
            self.label_distr = np.zeros(self.num_classes, dtype=int)

            for idx, k in enumerate(self.dict_distr.keys()):
                local_distr = np.array(self.dict_distr[k])
                label_exist_flag = np.where(local_distr > 0, 1, 0)
                self.label_distr += label_exist_flag
                self.local_vol[idx] = sum(local_distr)
                
                if self.global_distr is None:
                    self.global_distr = local_distr
                else:
                    self.global_distr += local_distr

            self.global_vol = self.local_vol.sum()

            '''
            self.dict_distr -> key: int, val: list
            '''

    def get_localLIBI(self, cid):
        '''
        LIBI: Label ImBalance Indicator
        '''
        distr = self.dict_distr[cid]
        if min(distr) == 0:
            ret = 'max/min = {} / 0'.format(max(distr))
        else:
            ret = str(max(distr)/min(distr))
        
        return ret

    def get_globalVIBI(self) -> float:
        '''
        VIBI: Volumn ImBalance Indicator
        '''
        return '{} = {}/{}'.format(self.local_vol.max()/self.local_vol.min(),
                                   self.local_vol.max(),
                                   self.local_vol.min())

    def get_globalLIBI(self) -> float:
        '''
        LIBI: Label ImBalance Indicator
        '''
        return '{} = {}/{}'.format(self.global_distr.max()/self.global_distr.min(),
                                   self.global_distr.max(),
                                   self.global_distr.min())
        # min could not be zero, otherwise num_classes should decrease

    def get_datasetInfo(self):
        print('Num of classes:', len(self.global_distr))
        print('Num of samples in each class:', self.global_distr)
        
        return self.global_distr

    def get_labelDistr(self):
        '''
        num of label habitation (owned by ? clients)
        inidicate the scarcity of label
        '''

        return self.label_distr
    
    def get_local_vol_frac(self):
        return self.local_vol / float(self.global_vol)

    def plot_local_distr_h(self, n_shard, title=False, legend=False):
        '''
        stacked bar graph ver. horizontal
        '''
        fig = plt.figure(figsize=(3, 15))
        if title:
            plt.title('Local Data Distribution - shard {}'.format(n_shard), y=1.1)
        plt.ylabel('Client ID')
        plt.xlabel('Data Volumn per Class')

        base = np.zeros(self.num_users, dtype=int)
        for c in range(self.num_classes):
            c_vol = np.array([self.dict_distr[x][c] for x in range(self.num_users)])
            plt.barh(range(self.num_users), c_vol,
                    left=base, label='class {}'.format(c))
            base += c_vol
        if legend:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    fancybox=True, shadow=True, ncol=5)
        return fig
    
    def plot_local_distr(self, n_shard, title=False, legend=False, color=True):
        '''
        stacked bar graph
        '''
        fig = plt.figure(figsize=(20, 6))
        plt.rcParams['font.size'] = 22

        if title:
            plt.title('Local Data Distribution - {} - shard {}'.format(self.filename, n_shard), y=1.1)
        plt.xlabel('Client ID')
        plt.ylabel('Data Volumn per Class')

        base = np.zeros(self.num_users, dtype=int)
        base_local_cls = np.zeros(self.num_users, dtype=int)

        for c in range(self.num_classes):
            c_vol = np.array([self.dict_distr[x][c] for x in range(self.num_users)])
            if color:
                plt.bar(range(self.num_users), c_vol,
                        bottom=base, label='class {}'.format(c))
            else:
                plt.bar(range(self.num_users), c_vol, color='grey',
                        bottom=base, label='class {}'.format(c))
            base += c_vol
            base_local_cls += np.where(c_vol > 0, 1, 0)
        if legend:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    fancybox=True, shadow=True, ncol=5)
        
        return fig, base_local_cls.mean()
    
    def get_vol_asc_ID(self):
        # map new ID:old ID [(99, 1), (2, 2), ..., (50, 1200)]
        sorted_vol = sorted(enumerate(self.local_vol), key=lambda x:x[1])
        vol_asc_ID = list(list(zip(*sorted_vol))[0])
        return vol_asc_ID
    
    def plot_sorted_local_distr(self, n_shard, title=False, legend=False, color=True, save=False):
        '''
        stacked bar graph
        '''
        fig, ax = plt.subplots(figsize=(16*0.3,9*0.3))
        #plt.rcParams['font.size'] = 22

        ax.set_xlabel('Client ID')
        ax.set_ylabel('Data Volumn per Class')

        vol_asc_ID = self.get_vol_asc_ID()

        base = np.zeros(self.num_users, dtype=int)
        base_local_cls = np.zeros(self.num_users, dtype=int)

        for c in range(self.num_classes):
            c_vol = np.array([self.dict_distr[x][c] for x in vol_asc_ID])
            if color:
                ax.bar(range(self.num_users), c_vol,
                        bottom=base, label='{}'.format(c))
            else:
                ax.bar(range(self.num_users), c_vol, color='grey',
                        bottom=base, label='{}'.format(c))
            base += c_vol
            base_local_cls += np.where(c_vol > 0, 1, 0)
        if legend:
            ax.legend(title='Class', fontsize=8, ncol=2, frameon=False)
            #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            #        fancybox=True, shadow=True, ncol=5)
        
        if title:
            ax.set_title('Local Data Distribution - {} - shard {}'.format(self.filename, n_shard), y=1.1)
        
        
        return fig, ax, base_local_cls.mean()

    def __repr__(self):
        '''
        print
        '''
        pass