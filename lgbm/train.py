import numpy as np
import torch
from config import CFG, logging_conf
from lgbm.datasets import load_dataframe, feature_engineering, custom_train_test_split, make_dataset
from lgbm.utils import class2dict, get_logger
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pandas as pd
if CFG.user_wandb:
    import wandb
    wandb.init(**CFG.wandb_kwargs, config=class2dict(CFG))


logger = get_logger(logging_conf)
use_cuda = torch.cuda.is_available() and CFG.use_cuda_if_available
device = torch.device("cuda" if use_cuda else "cpu")
print(device)


def main():
    logger.info("Task Started")

    logger.info("Data Preparing - Start")
    # train_data = load_dataframe(CFG.basepath, "cv_train_data.csv")
    # train_data = feature_engineering(train_data)
    train_data = pd.read_pickle('/opt/ml/level2-dkt-level2-recsys-08/data_pkl/all.pkl')
    train, valid = custom_train_test_split(train_data)
    FEATS, y_train, train, y_test, test = make_dataset(train, valid)

    
    cat_cols = ['i_head', 'i_mid','i_tail', 'hour', 'dow']
    cont_cols = [                        
            'user_correct_answer',
            'user_total_answer',
            'user_acc',            
            't_elapsed',            
            'cum_correct',
            # 'last_problem',
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
            'assessment_mean',
            'assessment_sum',
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
            'tail_elapsed_x']

    FEATS = cat_cols + cont_cols    

    lgb_train, lgb_test = lgb.Dataset(train[FEATS], y_train), lgb.Dataset(test[FEATS], y_test)

    logger.info("Data Preparing - Done")

    logger.info("Model Training - Start")
    model = lgb.train(
        {
            'objective': 'binary',
            # 'max_depth' : 13,
            'num_leaves' : 16,
            'metric' : 'binary_logloss',        
            # 'boosting' : 'dart'
        },
        lgb_train,
        valid_sets=[lgb_train, lgb_test],
        verbose_eval=100,
        num_boost_round = 1500,
        early_stopping_rounds=100,
        categorical_feature = cat_cols,
        
    )

    if CFG.user_wandb:
        wandb.watch(model)

    preds = model.predict(test[FEATS])
    acc = accuracy_score(y_test, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_test, preds)
    print(f'VALID AUC : {auc} ACC : {acc}\n')
    
    logger.info("Model Training - Done")

    # 학습된 모델 저장
    model.save_model("model.txt")

    logger.info("Task Complete")


if __name__ == "__main__":
    main()
