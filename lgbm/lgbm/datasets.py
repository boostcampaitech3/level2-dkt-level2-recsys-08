import os

import pandas as pd
import random
import lightgbm as lgb

dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}

def load_dataframe(basepath, filename):
    return pd.read_csv(os.path.join(basepath, filename), dtype=dtype, parse_dates=['Timestamp'])


def feature_engineering(df):
    
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)
    
    #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
    

    # diff()를 이용하여 시간 차이를 구해줍니다
    diff = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
    
    # 만약 0초만에 풀었으면 0으로 치환
    diff = diff.fillna(pd.Timedelta(seconds=0))
    
    # 시간을 전부 초단위로 변경합니다.
    diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

    # df에 elapsed(문제 풀이 시간)을 추가해줍니다.
    df['t_elapsed'] = diff
    
    # 문제 풀이 시간이 650초 이상은 이상치로 판단하고 제거합니다.
    df['t_elapsed'] = df['t_elapsed'].apply(lambda x : x if x <650 else None)
    
    # 대분류(앞 세자리)
    df['i_head']=df['testId'].apply(lambda x : int(x[1:4])//10)

    # 중분류(중간 세자리)
    df['i_mid'] = df['testId'].apply(lambda x : int(x[-3:]))

    # 문제 번호(분류를 제외한)
    df['i_tail'] = df['assessmentItemID'].apply(lambda x : int(x[-3:]))
        
    # 만들어 놓은 피쳐 추가
    path = "/opt/ml/input/data"
    # 유저 피쳐    
    userID_feature = pd.read_csv(os.path.join(path, "feature/userID_feature.csv"), index_col= 0)
    
    # 시험지 피쳐
    testId_feature = pd.read_csv(os.path.join(path,"feature/testId_feature.csv"), index_col= 0)
    # 태그 피쳐
    knowLedgedTag_acc = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
    knowLedgedTag_acc.columns = ["tag_mean", 'tag_sum']
    
    df = pd.merge(df, userID_feature, on=['userID', 'i_head'], how="left")
    df = pd.merge(df, testId_feature, on=['testId'], how="left")
    df = pd.merge(df, knowLedgedTag_acc, on=['KnowledgeTag'], how="left")
    
    return df


def custom_train_test_split(df, ratio=0.7, split=True):
    users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
    random.seed(42)
    random.shuffle(users)
    
    max_train_data_len = ratio*len(df)
    sum_of_train_data = 0
    user_ids =[]

    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)


    train = df[df['userID'].isin(user_ids)]
    test = df[df['userID'].isin(user_ids) == False]

    # test데이터셋은 각 유저의 마지막 interaction만 추출
    test = test[test['userID'] != test['userID'].shift(-1)]
    return train, test


def make_dataset(train, test):
    # 사용할 Feature 설정
    FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
            'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']

    # 원하는 피쳐 취사 선택(inference 파일도 변경해야함)
    FEATS += [
            'userID',
            'assessmentItemID',
            'testId',
            'answerCode',
            'Timestamp',
            'KnowledgeTag',
            'user_correct_answer',
            'user_total_answer',
            'user_acc',
            't_elapsed',
            'i_head',
            'i_mid',
            'i_tail',
            'u_head_mean',
            'u_head_count',
            'u_head_elapsed',
            'i_mid_elapsed',
            'i_mid_mean',
            'i_mid_sum',
            'i_mid_count',
            'i_mid_tag_count',
            'tag_mean',
            'tag_sum']


    # X, y 값 분리
    y_train = train['answerCode']
    train = train.drop(['answerCode'], axis=1)

    y_test = test['answerCode']
    test = test.drop(['answerCode'], axis=1)

    return FEATS, y_train, train, y_test, test