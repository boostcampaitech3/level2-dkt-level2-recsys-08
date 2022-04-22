import numpy as np
import torch
from config import CFG, logging_conf
from lgbm.datasets import load_dataframe, feature_engineering, custom_train_test_split, make_dataset
from lgbm.utils import class2dict, get_logger
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

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
    train_data = load_dataframe(CFG.basepath, "train_data.csv")
    train_data = feature_engineering(train_data)
    train, valid = custom_train_test_split(train_data)
    FEATS, y_train, train, y_test, test = make_dataset(train, valid)

    lgb_train, lgb_test = lgb.Dataset(train[FEATS], y_train), lgb.Dataset(test[FEATS], y_test)

    logger.info("Data Preparing - Done")

    logger.info("Model Training - Start")
    model = lgb.train(
        {'objective': 'binary'}, 
        lgb_train,
        valid_sets=[lgb_train, lgb_test],
        verbose_eval=100,
        num_boost_round=500,
        early_stopping_rounds=100
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
