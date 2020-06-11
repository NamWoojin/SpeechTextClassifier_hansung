### 공통 요소 ###
import os
import math
import random
import numpy as np
from sklearn.model_selection import train_test_split
os.chdir("C:\Users\NAM WOO JIN\Downloads\SpeechTextClassifier_hansung")

def read_data(filename, encoding):
    """읽기 함수"""
    with open(filename, 'r', encoding=encoding) as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[0:]                 # txt 파일의 헤더(id document label)는 제외하기는 1:
    return data


def write_data_list(list, filename, encoding):
    """쓰기 함수"""
    with open(filename, 'w') as f:
        for item in list:
            f.write('%s\t%s\t%s\t%s\n' % (item[0], item[1], item[2], item[3]))


#########################################################################
### train_test_split() 함수를 이용하여 훈련데이터와 테스트데이터 분리 ###
data = read_data('story_morphed_1013.txt', encoding='cp949')
train, test, = train_test_split(data, test_size=0.1)

write_data_list(list=train, filename='train_story_morphed_1013.txt', encoding='cp949')
write_data_list(list=test, filename='test_story_morphed_1013.txt', encoding='cp949')
