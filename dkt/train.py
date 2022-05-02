import os
import pickle
import torch
import wandb
import pandas as pd
from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess, add_features, post_process
from dkt.utils import setSeeds


def main(args):
    wandb.login()

    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device    

    args = add_features(args)

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
<<<<<<< HEAD
    train_data = preprocess.get_train_data() 
=======

    # 여기서 말하는 train_data는 학생별 testId, assessmentItemID, KnowledgeTag, answerCode 값 리스트를 의미한다.
    train_data = preprocess.get_train_data()
>>>>>>> 296752926a6e89406b4e2364fa5c62e99f6be183

    # (real) train-validation split
    train_data, valid_data = preprocess.split_data(train_data)   # 학생, 7:3

    # train_data = post_process(train_data, args)
    # valid_data = post_process(valid_data, args)

    # wandb.init(project="dkt", config=vars(args))
    trainer.run(args, train_data, valid_data)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
