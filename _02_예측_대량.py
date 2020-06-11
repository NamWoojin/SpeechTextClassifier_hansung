print('\n######## Environment Setting ########')


import os
from keras.models import load_model
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from keras.preprocessing.sequence import pad_sequences
import rhinoMorph3 as rhinoMorph
import pickle


path = 'C:/Users/lingu/OneDrive/Python Projects/SpeechTextClassifier_hansung'
test_file_name = 'test_story_morphed_1013.txt'              # 입력할 파일의 이름
filename = 'train_story_morphed_1013'
model_name = filename+'.h5'                                 # 분류 모델
tokenizer_name = filename+'.pickle'                         # 토크나이저

# 형태소 분석기 기동
rn = rhinoMorph.startRhino()


print('\n######## Load Model ########')
os.chdir(path+"/model")                                              # 현재 파일이 있는 폴더를 가져온다. 폴더에 한글이 있으면 인식하지 못하니 주의.
print("Current Directory:", os.getcwd())

loaded_model = load_model(model_name)

with open(tokenizer_name, 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)


def read_data(filename, encoding):
    """읽기 함수"""
    with open(filename, 'r', encoding=encoding) as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[0:]                 # txt 파일의 헤더(id document label)는 제외하기는 1:
    return data



print('\n######## Test Data Loading ########')
os.chdir(path)
data = read_data(test_file_name, encoding='cp949')      # 0: id, 1: 원본문, 2: 형태소분석본문, 3: 긍부정
texts = [line[1] for line in data]                      # 형태소분석되지 않은 본문
labels = [line[3] for line in data]                     # 긍부정 부분
labels_list = list(map(int, labels))                    # 문자열을 숫자형으로 변환


# 형태소 분석된 문장 샘플 보기
sample_data = rhinoMorph.onlyMorph_list(rn, texts[0], pos=['NNG', 'NNP', 'NP', 'VV', 'VA', 'XR', 'VCN', 'MAG', 'MAJ', 'IC', 'JKV', 'EF', 'SF'])
print('sample data:', sample_data)
print('joined sample data:', ' '.join(sample_data))


# 전체 문장 형태소 분석
morphed_texts = []
for data_each in texts:
    morphed_data_each = rhinoMorph.onlyMorph_list(rn, data_each, pos=['NNG', 'NNP', 'NP', 'VV', 'VA', 'XR', 'VCN', 'MAG', 'MAJ', 'IC', 'JKV', 'EF', 'SF'])
    joined_data_each = ' '.join(morphed_data_each)               # 문자열을 하나로 연결
    if joined_data_each:                                         # 내용이 있는 경우만 저장함
        morphed_texts.append(joined_data_each)

print('sample data:', morphed_texts[0])                          # 형태소 분석된 문장 샘플 보기


count = 0
result_list = []
idx = 0
for m_text in morphed_texts:
    m_text_split = m_text.split()
    print(idx, ':', m_text)                               # m_text와 morphed_texts[idx]는 동일하다

    data = loaded_tokenizer.texts_to_sequences([m_text])  # 토크나이징. 문자열을 word_index의 숫자 리스트로 변환
    x_test = pad_sequences(data, maxlen=100)              # 패딩. 길이를 고정시킨다.
    predictions = loaded_model.predict(x_test)            # 예측
    result_list.append(np.argmax(predictions))            # 확률값을 정수 형태로 변환
    idx = idx+1


# labels_list가 정답값이고, result_list가 예측값이다
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print("정확도:", round(accuracy_score(labels_list, result_list), 3))   # 앞에서 본 정확도와 값이 같음

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
precision = precision_score(labels_list, result_list, average="macro")
print("precision:", precision)
recall = recall_score(labels_list, result_list, average="macro")
print("recall:", recall)
f1_score = f1_score(labels_list, result_list, average="macro")
print("f1 score:", f1_score)

# 분류별 정확도
results = confusion_matrix(labels_list, result_list)
print("오차행렬:\n", results)
print('분류 0의 정확도: {}%'.format(round(results[0][0] / sum(results[0]) * 100), 3))
print('분류 1의 정확도: {}%'.format(round(results[1][1] / sum(results[1]) * 100), 3))
print('분류 2의 정확도: {}%'.format(round(results[2][2] / sum(results[2]) * 100), 3))
print('rule based count:', count)
