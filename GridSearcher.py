import pickle
import gc; gc.enable()
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.sparse import hstack, vstack, csr_matrix
from copy import deepcopy as cp
from sklearn.preprocessing import normalize

from wordbatch.models import FM_FTRL
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

'''
data_type:

tf-idf: text only
cv: text only
stage1
stage2
'''
class data_loader():
    def __init__(self, data_type='tf-idf', is_regression=True, is_train=True, is_pure=False):
        self.data_type=data_type
        self.is_regression=is_regression
        self.is_train=is_train
        self.is_pure = is_pure
        
    def load_oof_data(self):
        target_folder = 'text pred data/'
        target_prefix = ['text_lgb', 'text_fm', 'text_rg',
                        'seqno_rg', 'seqno_price_img_rg', 'seqno_img_rg', 
                        'price_rg', 'price_seqno_rg', 'price_img_rg', 'img1_rg',
                        'seqno_lr', 'seqno_price_img_lr', 'seqno_img_lr', 
                        'price_lr', 'price_seqno_lr', 'price_img_lr', 'img1_lr',
                        'all_rg']

        res = pd.DataFrame()
        for prefix in target_prefix:
            train_f = target_folder + prefix + '_oof_val_pred.csv'
            test_f = target_folder + prefix + '_oof_test_pred.csv'
            
            print("Loading: ", train_f, test_f)
            temp = pd.concat([pd.read_csv(train_f), pd.read_csv(test_f)]).reset_index(drop=True) 
            col = temp.columns.tolist()[0]
            res.loc[:, col] = temp[col].values
            
            del temp; gc.collect()
        
        return res
    
    def load_tf_idf_features(self):
        with open('title_text_feature_count_vec.pickle', 'rb') as handle:
            title_feature = pickle.load(handle)
            old_title_feature = title_feature
            title_feature = normalize(title_feature, norm='l2', axis=0)
            del old_title_feature; gc.collect()

            print('title text features loaded')

        # redundant with cat feature
        with open('param_text_feature_count_vec.pickle', 'rb') as handle:
            param_feature = pickle.load(handle)            
            old_param_feature = param_feature
            param_feature = normalize(param_feature, norm='l2', axis=0)
            del old_param_feature; gc.collect()                
            print('param text features loaded')
            
        with open('desc_text_feature_tf_vec.pickle', 'rb') as handle:
            text_feature = pickle.load(handle)
            print('desc text features loaded')

        res = hstack([title_feature, param_feature, text_feature]).tocsr()
        del title_feature, param_feature, text_feature; gc.collect()
        return res
    
    def load_char_wb_features(self):
        
        
        with open('title_text_feature_count_vec.pickle', 'rb') as handle:
            title_feature = pickle.load(handle)
            old_title_feature = title_feature
            title_feature = normalize(title_feature, norm='l2', axis=0)
            del old_title_feature; gc.collect()
            print('title text features loaded')
        
        # redundant with cat feature
        with open('param_text_feature_count_vec.pickle', 'rb') as handle:
            param_feature = pickle.load(handle)            
            old_param_feature = param_feature
            param_feature = normalize(param_feature, norm='l2', axis=0)
            del old_param_feature; gc.collect()             
            print('param text features loaded')
        
        with open('0530_CV_CHAR_WB_NGRAM_24/text_features_all.pickle', 'rb') as handle:
            text_feature = csr_matrix(pickle.load(handle), dtype=float)
            print('desc text features loaded')

        res = hstack([title_feature, param_feature, text_feature]).tocsr()
        del title_feature, param_feature, text_feature; gc.collect()
        return res
        
    def load_dense_features(self):
        data=None
        
        with open('train_features', 'rb') as handle:
            x_train = pickle.load(handle)

        with open('test_features', 'rb') as handle:
            x_test = pickle.load(handle)

        all_features = vstack([x_train, x_test]).tocsr()
        
        if not self.is_pure:
            oof_feat = self.load_oof_data()
            data = hstack([all_features, csr_matrix(oof_feat)]).tocsr()
            del all_features, oof_feat; gc.collect()
        else:
            data = all_features    
        
        return data
    
    def load_sparse_features(self):
        data=None
        
        with open('train_ohe_norm_features', 'rb') as handle:
            x_train = pickle.load(handle)

        with open('test_ohe_norm_features', 'rb') as handle:
            x_test = pickle.load(handle)

        all_features = vstack([x_train, x_test]).tocsr()
        
        if not self.is_pure:
            oof_feat = self.load_oof_data()
            data = hstack([all_features, csr_matrix(oof_feat)]).tocsr()
            del all_features, oof_feat; gc.collect()
        else:
            data = all_features    
        
        return data
    
    def load(self):
        data_type=self.data_type
        is_regression=self.is_regression
        is_train=self.is_train
        print('Arguments:', data_type, is_regression, is_train)
        
        if is_regression:
            train_y = pd.read_csv("regression_target.csv")
        else:
            train_y = pd.read_csv("classification_target.csv")

        train_y = train_y['deal_probability'].values
        train_len = train_y.shape[0]
        print('target loaded')
        
        if data_type == 'tf-idf':
            data = self.load_tf_idf_features()
        elif data_type == 'char_wb':
            data = self.load_char_wb_features()
            if is_train:
                return data, train_y
            else:
                return data
        elif data_type == 'dense':
            data = self.load_dense_features()
        elif data_type == 'sparse':
            data = self.load_sparse_features()
        else:
            return None

        if is_train:
            if (type(data) == csr_matrix) or (type(data) == np.ndarray):
                train_X = data[:train_len]
            else:
                train_X = data.loc[:train_len-1,:]
                
            del data; gc.collect()
            return train_X, train_y
        else:
            if (type(data) == csr_matrix) or (type(data) == np.ndarray):
                test_X = data[train_len:]
            else:
                test_X = data.loc[train_len:,:].reset_index(drop=True)
                
            del data; gc.collect()
            return test_X
    
def clip_rmse(ground_truth, predictions):
    predictions = np.clip(predictions, 0., 1.)
    return mean_squared_error(ground_truth, predictions)**.5

class model_loader():
    def __init__(self, model_type='lgb'):
        self.model_type = model_type
        
    def load(self, params):
        model_type = self.model_type
        _model_dict = {
            'lgb': lgb.LGBMRegressor,
            'fm': FM_FTRL,
            'rg': Ridge,
            'lr': LogisticRegression,
            'knn': KNeighborsRegressor,
            'mlp': MLPRegressor,
        }
        if model_type in _model_dict:
            return _model_dict[model_type](**params)
            
        print('Model type unspecified.')
        return None
        
def get_oof_predictions(X, y, test_X, model_loader, param, seed=719, fit_params=None, use_eval_set=False, predict_proba=False):
    kf = KFold(5, shuffle=True, random_state=seed)
    
    ret = np.zeros((len(y),))
    ret_test = np.zeros((test_X.shape[0],))
    ret_models = []
    counter = 1
    for train_index, val_index in kf.split(X):
        if (type(X) == csr_matrix) or (type(X) == np.ndarray):
            tr_X = X[train_index,:]
            val_X = X[val_index,:]
        else:
            tr_X = X.loc[train_index,:].reset_index(drop=True)
            val_X = X.loc[val_index,:].reset_index(drop=True)
                
        if type(y) == pd.Series:
            y = y.values
        tr_y = y[train_index]   
        val_y = y[val_index]  

        model = model_loader.load(param)
        if fit_params is not None:
            if use_eval_set is True:
                #print(fit_params)
                model.fit(tr_X, tr_y, 
                          eval_set=[(tr_X, tr_y), (val_X, val_y)], 
                          eval_names=['train', 'valid'], 
                          **fit_params)
            else:
                model.fit(tr_X, tr_y, **fit_params)
        else:
            model.fit(tr_X, tr_y)
                
        if predict_proba:
            ret[val_index] = model.predict_proba(val_X)[:,1]
            ret_test += model.predict_proba(test_X)[:,1]
        else:
            ret[val_index] = model.predict(val_X)
            ret_test += model.predict(test_X)
        
        ret_models.append(model)
        
        print('Fold', counter, 'completed.')
        counter += 1

    ret_test = ret_test/5
    return ret, ret_test, ret_models

def fit_params(X, y, model_loader, default_params, try_params, use_eval_set=False,
               fit_params=None, seed=719, loss_func=clip_rmse, predict_proba=False):
    
    params_list = ParameterGrid(try_params)
    
    kf = KFold(5, shuffle=True, random_state=seed)
    res_df = pd.DataFrame()
    best_param = None
    best_loss = None
    
    for i, param in enumerate(params_list):
        used_params = cp(default_params)
        used_params.update(param)
        
        losses = []
        for train_index, val_index in kf.split(X):
            #print(train_index)
            if (type(X) == csr_matrix) or (type(X) == np.ndarray):
                tr_X = X[train_index,:]
                val_X = X[val_index,:]
            else:
                tr_X = X.loc[train_index,:].reset_index(drop=True)
                val_X = X.loc[val_index,:].reset_index(drop=True)
                
            if type(y) == pd.Series:
                y = y.values
            tr_y = y[train_index]   
            val_y = y[val_index]  

            model = model_loader.load(used_params)
            if fit_params is not None:
                if use_eval_set is True:
                    model.fit(tr_X, tr_y, 
                          eval_set=[(tr_X, tr_y), (val_X, val_y)], 
                          eval_names=['train', 'valid'], 
                          **fit_params)
                else:
                    model.fit(tr_X, tr_y, **fit_params)
            else:
                model.fit(tr_X, tr_y)
                
            if predict_proba:
                tr_pred = model.predict_proba(tr_X)[:,1]
                val_pred = model.predict_proba(val_X)[:,1]
            else:
                tr_pred = model.predict(tr_X)
                val_pred = model.predict(val_X)
            
            tr_loss = loss_func(tr_y, tr_pred)
            loss = loss_func(val_y, val_pred)
            losses.append(loss)
            print(str(param)+' train loss: {:.6f}, valid loss:{:.6f}, loss_diff:{:.6f}'.format(tr_loss, loss, loss-tr_loss))
        
            del tr_X, val_X, tr_y, val_y, tr_pred, val_pred; gc.collect()
            
        losses = np.array(losses)
        print('=================>'+str(param)+' loss:{:.6f}'.format(losses.mean()))
        if best_loss is None or losses.mean() < best_loss:
            best_loss = losses.mean()
            best_param = str(param)
        
        res_df.loc[i, 'param'] = str(param)
        res_df.loc[i, 'val_loss_mean'] = losses.mean()
        res_df.loc[i, 'val_loss_std'] = losses.std()
        
    print('Best params:', str(best_param), '\tbest loss:', best_loss)
    return res_df