# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")

# celery
from celery import Celery

# standard imports
from joblib import dump, load
import pandas as pd
import numpy as np
from numpy.random import RandomState
from time import time
import pprint
import joblib
import os


# sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

# catboost
from catboost import Pool, CatBoostClassifier, cv

# sampling
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import NearMiss

# hyperparameter optimization
import hyperopt
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

# missing value handling
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean',filler='NA'):
        self.strategy = strategy
        self.fill = filler

    def fit(self, X, y=None):
        if self.strategy in ['mean','median']:
            if not all(X.dtypes == np.number):
                raise ValueError('dtypes mismatch np.number dtype is \
                               required for '+ self.strategy)
        if self.strategy == 'mean':
            self.fill = X.mean()
        elif self.strategy == 'median':
            self.fill = X.median()
        elif self.strategy == 'mode':
            self.fill = X.mode().iloc[0]
        elif self.strategy == 'fill':
            if type(self.fill) is list and type(X) is pd.DataFrame:
                self.fill = dict([(cname, v) for cname,v in zip(X.columns, self.fill)])
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

# shorten feature names for more readable graphing
def short_names(row):
    if row['Role'] == 'feature':
        result = 'X' + str(row['count'])
    elif row['Role'] == 'target':
        result = 'Y'
    else:
        result = ''
    return pd.Series(result)

# Create the app and point to the queue implementation
app = Celery('unravelML',
             backend='redis://169.62.176.245:6379/0',
             broker='redis://169.62.176.245:6379/0')

# this task below creates a preprocessed view of the uploaded data
# before handing it off for model train and visualization
@app.task

def variable_role(file_name):
    excel_file = '/opt/notebooks/demo/uploads/' + file_name
    df = pd.read_excel(excel_file)

    # feature names
    fnames = list(df.iloc[0, :])
    # variable names
    vnames = list(df.columns)
    # variable types
    vtypes = list(df.iloc[1, :])
    # store into a df
    var_roles = pd.DataFrame({'Variable': vnames,
                              'Role': fnames, 
                              'Type': vtypes
                             })

    # cumulative count
    var_roles['count'] = var_roles.groupby('Role').cumcount() + 1

    # apply short names
    var_roles['Feature'] = var_roles.apply(short_names, axis=1) 
    
    # save the processed mapping to a pickle file
    var_roles.to_pickle('/opt/notebooks/demo/celery/processed/' + file_name.split('.')[0] + '.pkl')

# this task trains a fitted model with simple hyperparameter tuning
# it also saves important metrics for model evaluation and interpretability charting
@app.task
    
def train_model(file_name):
    # read in excel data and variable roles
    excel_file = '/opt/notebooks/demo/uploads/' + file_name
    var_roles = pd.read_pickle('/opt/notebooks/demo/celery/processed/' + file_name.split('.')[0] + '.pkl')
    df = pd.read_excel(excel_file)

    # convert to proper data frame
    data = df.iloc[2:].reset_index(drop=True).copy()

    # store feature, target, meta columns into lists
    m_columns = var_roles[var_roles.Role=='meta'].Variable.tolist()
    f_columns = var_roles[var_roles.Role=='feature'].Variable.tolist()
    t_column = var_roles[var_roles.Role=='target'].Variable.tolist()

    # set data types according to user specification for catboost model 
    convert_dict = {each: ('category' if var_roles[var_roles.Variable==each].Type.values=='cat' \
                                           else 'float64' if var_roles[var_roles.Variable==each].Type.values=='num' \
                                           else 'object') for each in data.columns}

    data = data.astype(convert_dict) 

    # check which columns have missing values
    null_value_stats = data.isnull().sum(axis=0)
    #null_value_stats[null_value_stats != 0]

    # handle missing values
    xdata = CustomImputer(strategy='fill', filler=[0, 'NA']).fit_transform(data)

    # create metas, features, target dataframes
    metas = xdata[m_columns]
    xs = xdata[f_columns]
    feature_dict = {each: var_roles[var_roles.Variable==each].Feature.values[0] for each in xs.columns}
    xs = xs.rename(index=str, columns=feature_dict) # use shorter names
    ys = xdata[t_column]
    target_map = {each: var_roles[var_roles.Variable==each].Feature.values[0] for each in ys.columns}
    ys = ys.rename(index=str, columns=target_map) # use Y for target

    # make a note of categorical features
    cat_features_index = [xs.columns.get_loc(col) for col in xs.select_dtypes(include=['category']).columns]
    cat_features_names = [col for col in xs.select_dtypes(include=['category']).columns]
    cat_mappings = pd.DataFrame(list(zip(cat_features_index, cat_features_names)), columns=['Index', 'Name'])
    cat_mappings.to_pickle('/opt/notebooks/demo/celery/processed/' + file_name.split('.')[0] + '_catsMap.pkl')

    
    # create a label mapping for multiclass classification 
    y_label_mapping = pd.DataFrame.from_dict(dict(enumerate(ys.Y.cat.categories)), orient='Index', columns=['Label'])
    y_label_mapping.to_pickle('/opt/notebooks/demo/celery/processed/' + file_name.split('.')[0] + '_labelsMap.pkl')
    
    # split the data
    # features: xs
    # ground truth: ys
    # meta info: metas

    x_train, x_test, y_train, y_test, m_train, m_test = train_test_split(xs, ys, metas,
                                                        test_size=0.20,
                                                        random_state=42,
                                                        stratify=ys)
    
    m_test.to_pickle('/opt/notebooks/demo/celery/processed/' + file_name.split('.')[0] + '_mTest.pkl')  
    x_test.to_pickle('/opt/notebooks/demo/celery/processed/' + file_name.split('.')[0] + '_xTest.pkl')
    y_test.to_pickle('/opt/notebooks/demo/celery/processed/' + file_name.split('.')[0] + '_yTest.pkl')
    
    # fix imbalance
    # SMOTE-NC handles categorical features
    smt = SMOTENC(categorical_features=cat_features_index)
    x_res, y_res = smt.fit_sample(x_train, np.ravel(y_train))
    
    res_convert_dict = {each: ('category' if var_roles[var_roles.Feature==each].Type.values=='cat' \
                                       else 'float64' if var_roles[var_roles.Feature==each].Type.values=='num' \
                                       else 'object') for each in var_roles[var_roles.Role=='feature'].Feature.values.tolist()}
    
    x_res_df = pd.DataFrame(data=x_res, columns=x_test.columns).astype(res_convert_dict) 
    y_res_df = pd.DataFrame(data=y_res, index=range(y_res.shape[0]), columns=['Y'])
    
    x_res_df.to_pickle('/opt/notebooks/demo/celery/processed/' + file_name.split('.')[0] + '_x_rTrain.pkl')
    y_res_df.to_pickle('/opt/notebooks/demo/celery/processed/' + file_name.split('.')[0] + '_y_rTrain.pkl')

    
    # set loss function and train/test pools
    ls_fnc = ("MultiClass" if len(ys.Y.unique()) > 2 else "CrossEntropy")
    train_pool = Pool(x_res, np.ravel(y_res), cat_features=cat_features_index)
    test_pool = Pool(x_test, np.ravel(y_test), cat_features=cat_features_index)

    # set up stratified k-fold CV
    def hyperopt_objective(params):
        # Initializing a CatBoostClassifier
        model = CatBoostClassifier(
            task_type = 'GPU',
            loss_function=ls_fnc,
            iterations=100,
            eval_metric='AUC',
            logging_level='Silent', 
            early_stopping_rounds=5,
        )

        cv_data = cv(
            train_pool,
            model.get_params(), 
            fold_count = 3,
            shuffle = True,
            stratified = True, 
        )

        best_acc = np.max(cv_data['test-AUC-mean'])
        
        # negate because hyperopt minimizes the objective
        return {'loss': -best_acc, 'status': hyperopt.STATUS_OK}

    # hyperparameter tuning 
    params_space = {
        'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 2, 1),
        'max_depth': scope.int(hyperopt.hp.quniform('max_depth', 6, 10, 1)),
        'iterations' : hyperopt.hp.choice('iterations', np.arange(1000, 10000, 100, dtype=int)),
        'learning_rate' : hyperopt.hp.quniform('learning_rate', 0.025, 0.5, 0.025),
#         'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),

    }

    trials = hyperopt.Trials()

    best = hyperopt.fmin(
        hyperopt_objective,
        space=params_space,
        algo=hyperopt.tpe.suggest,
        max_evals=5,
        trials=trials, 
        rstate=RandomState(42)
    )

    best_params = hyperopt.space_eval(params_space, best)
    
    # save best model
    tuned_model = CatBoostClassifier(
        task_type = 'GPU',
        loss_function=ls_fnc,
        l2_leaf_reg=int(best_params['l2_leaf_reg']),
        learning_rate=best_params['learning_rate'],
        max_depth=int(best_params['max_depth']),
        iterations=100,
        #iterations=best_params['iterations'],
        random_seed=42,
        logging_level='Silent', 
        early_stopping_rounds=5, 
    )

    tuned_model.fit(train_pool, plot=False)
    
    # save fitted model
    tuned_model.save_model('/opt/notebooks/demo/celery/processed/' + file_name.split('.')[0] + '_fitted.dump')
    
    # save hyper parameters
    pd.DataFrame.from_dict(tuned_model.get_params(), orient='index', columns=['Value']).to_pickle('/opt/notebooks/demo/celery/processed/' + file_name.split('.')[0] + '_params.pkl')

    predictions = tuned_model.predict(test_pool)
    predictions_p = tuned_model.predict(test_pool, prediction_type='Probability')
    preds_reshape = np.asarray(predictions.reshape(predictions.shape[0], 1))
    
    scored_test_pool = pd.concat([m_test.reset_index(drop=True), 
                                  y_test.reset_index(drop=True), 
                                  pd.DataFrame(preds_reshape, columns=['Pred_Y']), 
                                  pd.DataFrame(predictions_p, 
                                               columns=['PredProb_' + str(y_label_mapping.loc[each].Label) 
                                                        for each in y_label_mapping.index])], 
                                 axis=1)
    
    scored_test_pool.to_pickle('/opt/notebooks/demo/celery/processed/' + file_name.split('.')[0] + '_test_pred.pkl')
      
    