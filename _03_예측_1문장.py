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

filename = 'train_story_morphed_1013'
model_name = filename+'.h5'                                 # 분류 모델
tokenizer_name = filename+'.pickle'                         # 토크나이저


# 형태소 분석기 기동
rn = rhinoMorph.startRhino()


print('\n######## Load Model ########')
os.chdir(path+"/model")                                     # 현재 파일이 있는 폴더를 가져온다. 폴더에 한글이 있으면 인식하지 못하니 주의.
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



print('\n######## START ########')
os.chdir(path)

while (True):
    user_input = input("내용을 입력하세요: ")
    if (user_input == "Quit") or (user_input == "quit") or (user_input == "종료"):
        print("시스템을 종료합니다.")
        break

    # 형태소 분석
    morphed_input = rhinoMorph.onlyMorph_list(rn, user_input, pos=['NNG', 'NNP', 'NP', 'VV', 'VA', 'XR', 'VCN', 'MAG', 'MAJ', 'IC', 'JKV', 'EF', 'SF'])

    text = []
    text.append(user_input)
    text = rhinoMorph.onlyMorph_list(rn, text[0])           # 형태소 분석
    text = [' '.join(text)]

    data = loaded_tokenizer.texts_to_sequences(text)        # 토크나이징
    x_test = pad_sequences(data, maxlen=100)                # 패딩

    prediction = loaded_model.predict(x_test)
    final_prediction = np.argmax(prediction[0])


    # 최종 출력
    if final_prediction == 0:
        print("중립 감정입니다", prediction)
    elif final_prediction == 1:
        print("긍정 감정입니다", prediction)
    elif final_prediction == 2:
        print("부정 감정입니다", prediction)
    else:
        print("결과값 오류입니다")
