from sklearn import metrics
from sklearn.metrics import RocCurveDisplay, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from dataset import custom_train_test_split, make_dataset

from sklearn.metrics import RocCurveDisplay, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc


def get_metric(targets, preds):
    auc = roc_auc_score(targets, preds)
    acc = accuracy_score(targets, np.where(preds >= 0.5, 1, 0))
    precsion = precision_score(targets, np.where(preds >= 0.5, 1, 0))
    recall = recall_score(targets, np.where(preds >= 0.5, 1, 0))
    F1_score = f1_score(targets, np.where(preds >= 0.5, 1, 0))

    print('auc :',auc)
    print('acc :',acc)
    print('precision :',precsion)
    print('recall :',recall)

def test_to_csv(preds, name:str):
    
    result = []
    for n,i in enumerate(preds):
        row = {}    
        row['id'] = n
        row['prediction'] = i
        result.append(row)
    pd.DataFrame(result).to_csv(f'output/{name}.csv', index=None)

def main():
    cate_cols = [
                'assessmentItemID',
                'testId',
                'KnowledgeTag',
                'hour',
                'dow',
                # 'i_head',
                # 'i_mid',
                # 'i_tail',
    ]
    cont_cols = [                        
                'user_correct_answer',
                'user_total_answer',
                'user_acc',            
                't_elapsed',            
                'cum_correct',
                'last_problem',
                'head_term',
                # 'left_asymptote',
                'elo_prob',
                'pkt',
                'u_head_mean',
                'u_head_count',
                'u_head_std',
                'u_head_elapsed',
                'i_mid_elapsed',
                'i_mid_mean',
                'i_mid_std',
                'i_mid_sum',
                'i_mid_count',
                'i_mid_tag_count',
                # 'assessment_mean',
                # 'assessment_sum',
                # 'assessment_std',
                'tag_mean',
                'tag_sum',
                # 'tag_std',
                'tail_mean',
                'tail_sum',
                # 'tail_std',
                'hour_mean',
                'hour_sum',
                # 'hour_std',
                'dow_mean',
                'dow_sum',
                # 'dow_std',
                'tag_elapsed',
                'tag_elapsed_o',
                'tag_elapsed_x',
                'assessment_elapsed',
                'assessment_elapsed_o',
                'assessment_elapsed_x',
                'tail_elapsed',
                'tail_elapsed_o',
                'tail_elapsed_x'
                ]

    FEATS = cate_cols + cont_cols

    train_data = pd.read_pickle('/opt/ml/level2-dkt-level2-recsys-08/data_pkl/train_data.pkl')
    valid_user = pd.read_csv('/opt/ml/input/data/cv_valid_data.csv').userID.unique()
    from dataset import feature_engineering, custom_train_test_split, make_dataset


    train = train_data[train_data.userID.isin(valid_user)==False]
    valid = train_data[train_data.userID.isin(valid_user)==True]

    y_train, x_train, y_valid, x_valid = make_dataset(train, valid)

    test = pd.read_pickle('/opt/ml/level2-dkt-level2-recsys-08/data_pkl/test_data-1.pkl')

    train_pool = Pool(x_train[FEATS] ,y_train, cat_features = cate_cols)
    eval_pool = Pool(x_valid[FEATS] , y_valid, cat_features = cate_cols)

    # num_round? 1000~ 10000
    model1 = CatBoostClassifier(
            iterations = 1000,
            random_seed = 42,
            learning_rate = 0.01,
            loss_function = 'Logloss', 
            custom_metric = ['Logloss','AUC'],
            early_stopping_rounds = 30,
            use_best_model =  True,
            task_type = "GPU",
            bagging_temperature = 1,
            verbose = False)

    model2 = CatBoostClassifier(
            iterations = 3000,
            random_seed = 42,
            learning_rate = 0.005,
            loss_function = 'Logloss', 
            custom_metric = ['Logloss','AUC'],
            early_stopping_rounds = 30,
            use_best_model =  True,
            task_type = "GPU",
            bagging_temperature = 1,
            verbose = False)

    model3 = CatBoostClassifier(
            iterations = 5000,
            random_seed = 42,
            learning_rate = 0.001,
            loss_function = 'Logloss', 
            custom_metric = ['Logloss','AUC'],
            early_stopping_rounds = 30,
            use_best_model =  True,
            task_type = "GPU",
            bagging_temperature = 1,
            verbose = False)

    model4 = CatBoostClassifier(
            iterations = 4000,
            random_seed = 35,
            learning_rate = 0.003,
            loss_function = 'Logloss', 
            custom_metric = ['Logloss','AUC'],
            early_stopping_rounds = 30,
            use_best_model =  True,
            task_type = "GPU",
            bagging_temperature = 1,
            verbose = False)

    model5 = CatBoostClassifier(
            iterations = 4500,
            random_seed = 2020,
            learning_rate = 0.001,
            loss_function = 'Logloss', 
            custom_metric = ['Logloss','AUC'],
            early_stopping_rounds = 30,
            use_best_model =  True,
            task_type = "GPU",
            bagging_temperature = 1,
            verbose = False)

    model1.fit(train_pool, eval_set=eval_pool, plot=False)
    model2.fit(train_pool, eval_set=eval_pool, plot=False)
    model3.fit(train_pool, eval_set=eval_pool, plot=False)
    model4.fit(train_pool, eval_set=eval_pool, plot=False)
    model5.fit(train_pool, eval_set=eval_pool, plot=False)

    model1.save_model('/opt/ml/level2-dkt-level2-recsys-08/LetsEnsemble/model_save/catboost/model1.cbm')
    model2.save_model('/opt/ml/level2-dkt-level2-recsys-08/LetsEnsemble/model_save/catboost/model2.cbm')
    model3.save_model('/opt/ml/level2-dkt-level2-recsys-08/LetsEnsemble/model_save/catboost/model3.cbm')
    model4.save_model('/opt/ml/level2-dkt-level2-recsys-08/LetsEnsemble/model_save/catboost/model4.cbm')
    model5.save_model('/opt/ml/level2-dkt-level2-recsys-08/LetsEnsemble/model_save/catboost/model5.cbm')


    test_preds1 = model1.predict(test[FEATS], prediction_type='Probability')[:,1]
    test_preds2 = model2.predict(test[FEATS], prediction_type='Probability')[:,1]
    test_preds3 = model3.predict(test[FEATS], prediction_type='Probability')[:,1]
    test_preds4 = model4.predict(test[FEATS], prediction_type='Probability')[:,1]
    test_preds5 = model5.predict(test[FEATS], prediction_type='Probability')[:,1]

    valid_preds1 = model1.predict(x_valid[FEATS], prediction_type='Probability')[:,1]
    valid_preds2 = model2.predict(x_valid[FEATS], prediction_type='Probability')[:,1]
    valid_preds3 = model3.predict(x_valid[FEATS], prediction_type='Probability')[:,1]
    valid_preds4 = model4.predict(x_valid[FEATS], prediction_type='Probability')[:,1]
    valid_preds5 = model5.predict(x_valid[FEATS], prediction_type='Probability')[:,1]


    # print('Fold no: {}'.format(fold_))
    print("AUC LGB1:{} ".format(get_metric(y_valid, valid_preds1)))
    print("AUC LGB2:{} ".format(get_metric(y_valid, valid_preds2)))
    print("AUC LGB3:{} ".format(get_metric(y_valid, valid_preds3)))
    print("AUC LGB4:{} ".format(get_metric(y_valid, valid_preds4)))
    print("AUC LGB5:{} ".format(get_metric(y_valid, valid_preds5))) 

    new_valid = x_valid[FEATS].copy()
    new_valid.loc[:,'predict1'] = valid_preds1
    new_valid.loc[:,'predict2'] = valid_preds2
    new_valid.loc[:,'predict3'] = valid_preds3
    new_valid.loc[:,'predict4'] = valid_preds4
    new_valid.loc[:,'predict5'] = valid_preds5

    valid_tail = new_valid[new_valid.index.isin(x_valid.groupby('userID').tail(1).index)==True]
    new_valid = new_valid[new_valid.index.isin(x_valid.groupby('userID').tail(1).index)==False]

    new_test = test[FEATS].copy()
    new_test.loc[:,'predict1'] = test_preds1
    new_test.loc[:,'predict2'] = test_preds2
    new_test.loc[:,'predict3'] = test_preds3
    new_test.loc[:,'predict4'] = test_preds4
    new_test.loc[:,'predict5'] = test_preds5

    FEATS += [
            'predict1',
            'predict2'
            'predict3'
            'predict4'
            'predict5'
                ]  
    
    y_tail = y_valid[y_valid.index.isin(x_valid.groupby('userID').tail(1).index)==True]
    y_new_valid = y_valid[y_valid.index.isin(x_valid.groupby('userID').tail(1).index)==False]

    train_pool = Pool(new_valid[FEATS] ,y_new_valid, cat_features = cate_cols)
    eval_pool = Pool(valid_tail[FEATS] , y_tail, cat_features = cate_cols)


    Final_cat = CatBoostClassifier(
                iterations = 500,
                random_seed = 42,
                learning_rate = 0.002,
                loss_function = 'Logloss', 
                custom_metric = ['Logloss','AUC'],
                early_stopping_rounds = 30,
                use_best_model =  True,
                task_type = "GPU",
                bagging_temperature = 1,
                verbose = False)

    Final_cat.fit(train_pool, eval_set=eval_pool, plot=True)

    Final_valid_preds = Final_cat.predict(valid_tail, prediction_type='Probability')[:,1]
    Final_test_preds = Final_cat.predict(new_test, prediction_type='Probability')[:,1]


    # print('Fold no: {}'.format(fold_))
    get_metric(y_tail, Final_valid_preds)

    from datetime import date, datetime, timezone, timedelta

    KST = timezone(timedelta(hours=9))
    time_record = datetime.now(KST)
    _day = str(time_record)[:10]
    _time = str(time_record.time())[:8]
    now_time = _day+'_'+_time

    test_to_csv(Final_test_preds,f'belnding_catboost_{now_time}')

if __name__ == "__main__":
    main()