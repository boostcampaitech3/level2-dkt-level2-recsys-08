import os

import torch
from config import CFG, logging_conf
from lgbm.datasets import load_dataframe, feature_engineering
from lgbm.utils import get_logger
import lightgbm as lgb


logger = get_logger(logging_conf)
use_cuda = torch.cuda.is_available() and CFG.use_cuda_if_available
device = torch.device("cuda" if use_cuda else "cpu")

if not os.path.exists(CFG.output_dir):
    os.makedirs(CFG.output_dir)


def main():
    logger.info("Task Started")

    logger.info("Data Preparing - Start")

    test_data = load_dataframe(CFG.basepath, "test_data.csv")
    test_data = feature_engineering(test_data)
    # test 데이터셋은 각 유저의 마지막 interaction만 추출
    test_data = test_data[test_data['userID'] != test_data['userID'].shift(-1)]
    test_data = test_data.drop(['answerCode'], axis=1)

    logger.info("Data Preparing - Done")



    logger.info("Inference - Start")

    # 학습된 모델 불러오기
    # 참고 : https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py#L82-L84
    model = lgb.Booster(model_file="model.txt")
    
    # 사용할 Feature 설정
    FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
            'user_acc', 'i_mid_sum', 'tag_mean','tag_sum']

    # 원하는 피쳐 취사 선택(Dataset.py 파일도 변경해야함)
    FEATS += [ 
            'u_head_mean',
            'i_mid_mean'                                                                                                  
            't_elapsed',            
            'u_head_elapsed',
            'i_mid_elapsed',           
            ]



    total_preds = model.predict(test_data[FEATS])
    logger.info("Inference - Done")


    logger.info("Save Output - Start")

    output_dir = 'output/'
    write_path = os.path.join(output_dir, "submission.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))

    logger.info("Save Output - Done")


    logger.info("Task Complete")


if __name__ == "__main__":
    main()
