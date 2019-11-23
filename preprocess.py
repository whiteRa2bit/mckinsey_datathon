import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler



def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)






class TargetEncoder:
    def __init__(self, cat_names, use_noise=False, noise_sigma=0.1, use_smoothing=False, smoothing_coef=10):
        self.cat_names = cat_names
        self.use_noise = use_noise
        self.noise_sigma = noise_sigma
        self.use_smoothing = use_smoothing
        self.smoothing_coef = smoothing_coef
        
        
    def fit(self, X, y):
        feature_values_dict = {feature: {} for feature in self.cat_names}
        for feature in self.cat_names:
            feature_values_dict[feature] = {key: -1 for key in X[feature].unique()}
            for feature_value in feature_values_dict[feature].keys():
                idxs = X[X[feature] == feature_value].index
                
                if not self.use_smoothing:
                    feature_values_dict[feature][feature_value] = np.mean(y[idxs])
                else:
                    feature_values_dict[feature][feature_value] = (np.sum(y[idxs]) + \
                                                self.smoothing_coef * np.mean(y)) / (len(X) + self.smoothing_coef)

        self.feature_values_dict = feature_values_dict
        return self
    
    def transform(self, X_in):
        X = X_in.copy(deep=True)
        
        try:
            self.feature_values_dict
        except:
            raise AttributeError('Before applying transform method fit should be called')
            
        for feature in self.cat_names:
            cur_series = X[feature]
            new_series = []
            for value in cur_series:
                new_series.append(self.feature_values_dict[feature][value] + \
                                                self.use_noise * np.random.normal(scale=self.noise_sigma))

            X[feature] = np.array(new_series)
            
        return X         
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    
class CustomOhe:
    def __init__(self, cat_mask):
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.cat_mask = cat_mask
        pass
    
    def fit(self, X):
        X_cat = X[X.columns[self.cat_mask]]
        X_num = X[X.columns[~self.cat_mask]]
        
        self.ohe.fit(X_cat)  
        return self
    
    def transform(self, X):
        X_cat = X[X.columns[self.cat_mask]]
        X_num = X[X.columns[~self.cat_mask]]
        
        X_cat_ohe = self.ohe.transform(X_cat).toarray()

        return np.hstack((X_cat_ohe, X_num))
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
  
    
def preprocess(df_train_in, df_test_in, use_kaggle_target_encoding=False, use_custom_target_encoding=False,\
               use_ohe=False, use_scaling=True, filter_features=True):
    df_train = df_train_in.copy()
    df_test = df_test_in.copy()
    
    df_train.fillna(0, inplace=True)
    df_train.drop(columns=['building_id'], inplace=True)
    df_test.drop(columns=['building_id'], inplace=True)
    
    X_train = df_train.drop(columns=['target'])
    X_test = df_test.drop(columns=['target'])
    y_train = df_train['target']

    if filter_features:
        valuable_features = []

        for feature in X_train.columns:
            if feature[:16] in ['has_geotechnical', 'has_superstructu', 'height_ft_pre_eq']:
                continue
            else:
                valuable_features.append(feature)
                
        X_train = X_train[valuable_features]
        X_test = X_test[valuable_features]
        
        
    cat_features = []
    num_features = []
    for i, feature_type in enumerate(df_train.dtypes):
        if feature_type == 'object':
            cat_features.append(df_train.dtypes.index[i])
        else:
            num_features.append(df_train.dtypes.index[i])

            
    if use_custom_target_encoding and use_ohe:
        raise AttributeError("Choose only one option to deal with categorical features")

    if use_kaggle_target_encoding:
        for feature in cat_features:
            X_train[feature], X_test[feature] = target_encode(X_train[feature], X_test[feature], y_train)
          
    if use_custom_target_encoding:
        target_encoder = TargetEncoder(cat_features, use_noise=True, use_smoothing=True)
        X_train = target_encoder.fit_transform(X_train, y_train)
        X_test = target_encoder.transform(X_test)
        
    if use_ohe:
        cat_mask = np.array([True if feature in cat_features else False for feature in X_train.columns])
        ohe = CustomOhe(cat_mask)
        X_train = ohe.fit_transform(X_train)
        X_test = ohe.transform(X_test)
        
    if use_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train
