import os

os.chdir("C:/Users/lingu/OneDrive/Python Projects/SpeechTextClassifier_hansung/data")


def read_data(filename, encoding):
    """읽기 함수"""
    with open(filename, 'r', encoding=encoding) as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]                 # txt 파일의 헤더(id document label)는 제외하기는 1:
    return data


def write_data(data, filename, encoding):
    """쓰기 함수"""
    with open(filename, 'w', encoding=encoding) as f:
        f.write(data)

data = read_data('포커스테마 세계명작동화_all_1013.txt', encoding='cp949')        # 연습파일은 ratings_small.txt  0은 부정, 1은 긍정
print(len(data))                                                              # 11513
print(len(data[0]))                                                           # 3개의 컬럼으로 나뉘어 있다
print(data[0])


# 형태소 분석기 기동
import rhinoMorph3 as rhinoMorph
rn = rhinoMorph.startRhino()

# 형태소 분석된 문장 샘플 보기
sample_data = rhinoMorph.onlyMorph_list(rn, data[0][1], pos=['NNG', 'NNP', 'NP', 'VV', 'VA', 'XR', 'VCN', 'MAG', 'MAJ', 'IC', 'JKV', 'EC', 'EF', 'SF'])
print('sample data:', sample_data)
print('joined sample data:', ' '.join(sample_data))



# 전체 문장 형태소 분석
morphed_data = ''
for data_each in data:
    morphed_data_each = rhinoMorph.onlyMorph_list(rn, data_each[1], pos=['NNG', 'NNP', 'NP', 'VV', 'VA', 'XR', 'VCN', 'MAG', 'MAJ', 'IC', 'JKV', 'EF', 'SF'])
    joined_data_each = ' '.join(morphed_data_each)                                  # 문자열을 하나로 연결
    if joined_data_each:                                                            # 내용이 있는 경우만 저장함
        morphed_data += data_each[0]+"\t"+data_each[1]+"\t"+joined_data_each+"\t"+data_each[2]+"\n"


# 형태소 분석된 파일 저장
os.chdir("C:/Users/lingu/OneDrive/Python Projects/SpeechTextClassifier_hansung")
write_data(morphed_data, 'story_morphed_1013.txt', encoding='cp949')

