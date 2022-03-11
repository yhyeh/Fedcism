import numpy as np
import os
import pickle

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
            self.local_vol = np.zeros(len(self.dict_distr))
            self.label_distr = np.zeros(len(self.dict_distr[0]))

            for idx, k in enumerate(self.dict_distr.keys()):
                local_distr = np.array(self.dict_distr[k])
                label_exist_flag = np.where(local_distr > 0, 1, 0)
                self.label_distr += label_exist_flag
                self.local_vol[idx] = sum(local_distr)
                
                if self.global_distr is None:
                    self.global_distr = local_distr
                else:
                    self.global_distr += local_distr
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
        return '{} = {}/{}'.format(self.local_vol.max()/self.local_vol.min(), self.local_vol.max(), self.local_vol.min())

    def get_globalLIBI(self) -> float:
        '''
        LIBI: Label ImBalance Indicator
        '''
        return self.global_distr.max()/self.global_distr.min()
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
    
    def plot_local_distr(self):
        '''
        stacked bar graph
        '''
        pass
    def __repr__(self):
        '''
        print
        '''
        pass