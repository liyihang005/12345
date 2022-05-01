import numpy as np
import pandas as pd
import gensim
import string
import logging
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.data_utils import get_file
import jieba as jb
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier



class NewPre(object):

    def __init__(self):
        pass

    @staticmethod
    def get_main_complaints(raw_dataframe, cmp_counter_list, top_number, type_num):
        if type_num == 1:
            type_iloc = 3
        elif type_num == 2:
            type_iloc = 4
        elif type_num == 3:
            type_iloc = 5
        res_complaints = []
        res_cmps_tp = []
        needed_cmp_num_list = cmp_counter_list[:top_number]
        for i in range(raw_dataframe.shape[0]):
            if raw_dataframe.iloc[i, type_iloc] in needed_cmp_num_list:
                res_complaints.append(raw_dataframe.iloc[i, 2])
                res_cmps_tp.append(raw_dataframe.iloc[i, type_iloc])

        return res_complaints, res_cmps_tp

    @staticmethod
    def get_main_complaints_keras(raw_dataframe, cmp_counter_list, top_number, type_num):
        if type_num == 1:
            type_iloc = 3
        elif type_num == 2:
            type_iloc = 4
        elif type_num == 3:
            type_iloc = 5
        res_complaints = []
        res_cmps_tp = []
        needed_cmp_num_list = cmp_counter_list[:top_number]
        for i in range(raw_dataframe.shape[0]):
            if raw_dataframe.iloc[i, type_iloc] in needed_cmp_num_list:
                res_complaints.append(raw_dataframe.iloc[i, 2])
                res_cmps_tp.append(str(raw_dataframe.iloc[i, type_iloc]))

        return res_complaints, res_cmps_tp

    @staticmethod
    def read_data(file_path=r'D:\彭晓\LiYiHang\3WDatasetThreeType_simple.xlsx', sheet_name_=3):
        raw_df = pd.read_excel(file_path, sheet_name=sheet_name_)
        complaints = []
        for i in range(raw_df.shape[0]):
            complaints.append(raw_df.iloc[i, 2])

        complaints_type1 = []
        complaints_type2 = []
        complaints_type3 = []
        for i in range(raw_df.shape[0]):
            complaints_type1.append(raw_df.iloc[i, 3])
            complaints_type2.append(raw_df.iloc[i, 4])
            complaints_type3.append(raw_df.iloc[i, 5])

        return raw_df, complaints, complaints_type1, complaints_type2, complaints_type3

    @staticmethod
    def get_tp_star_name_list(complaints_type_star):
        type_star_distribution = Counter(complaints_type_star)
        return list(type_star_distribution.keys())

    @staticmethod
    def stopwordslist(filepath):
        stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
        return stopwords

    @staticmethod
    def rm_parenthesis(tmp):
        tmp = tmp.strip()
        start = 0
        end = 0
        for i in range(len(tmp)):
            if tmp[i] == "（" or tmp[i] == "(":
                start = i
                # print(tmp[i:])
                break
        for j in range(len(tmp)):
            if tmp[len(tmp) - 1 - j] == "）" or tmp[len(tmp) - 1 - j] == ")":
                end = len(tmp) - 1 - j
                break
        if start == end:
            return tmp
        if end < (len(tmp) - 1):
            return tmp[:start].strip() + tmp[end + 1:]
        else:
            return tmp[:start].strip()

    @staticmethod
    def seg_sentence(sentence, stopword_file=r'D:\彭晓\LiYiHang\stopwords-master\hit_stopwords.txt'):

        sentence_seged = jb.cut(NewPre.rm_parenthesis(sentence))
        # 修改为本机的 哈工大停用词表.txt地址
        stopwords = NewPre.stopwordslist(stopword_file)
        outstr = []
        for word in sentence_seged:
            if word not in stopwords:
                if re.match(r'[\u4e00-\u9fa5][\u4e00-\u9fa5]+$', word, flags=0) and not re.match(
                        r'(一)[\u4e00-\u9fa5]+$', word, flags=0):
                    outstr.append(word)
        return outstr

    @staticmethod
    def get_sentense_from_file(sentenses_list):
        sens = []
        max_len = 0
        for s in sentenses_list:
            tmp_sen = NewPre.seg_sentence(s)
            sens.append(tmp_sen)
            if len(tmp_sen) >= max_len:
                max_len = len(tmp_sen)

        return sens, max_len

    @staticmethod
    def word2idx(word):
        return word_model.wv.key_to_index[word]
        # return word_model.wv.vocab[word].index

    @staticmethod
    def idx2word(idx):
        return word_model.wv.index_to_key[idx]

    @staticmethod
    def word2vec_test():
        raw_data, r_data, cp1, cp2, cp3 = NewPre.read_data()
        tp_1_name_list = NewPre.get_tp_star_name_list(cp1)
        tp_2_name_list = NewPre.get_tp_star_name_list(cp2)
        tp_3_name_list = NewPre.get_tp_star_name_list(cp3)
        r_data_3, needed_cmps_tp3 = NewPre.get_main_complaints(raw_data, tp_3_name_list, 60, 3)
        r_data_2, needed_cmps_tp2 = NewPre.get_main_complaints(raw_data, tp_2_name_list, 60, 2)
        r_data_1, needed_cmps_tp1 = NewPre.get_main_complaints(raw_data, tp_1_name_list, 60, 1)

        run_data = r_data_3
        run_type = needed_cmps_tp3

        sentences, _ = NewPre.get_sentense_from_file(run_data)
        model = gensim.models.word2vec.Word2Vec(min_count=1)
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=100)
        res_array = []
        for s in sentences:
            tmp = np.zeros((len(s), len(model.wv[sentences[0][0]])), dtype=np.float32)
            for w_i in range(len(s)):
                tmp[w_i] = model.wv[s[w_i]]
            tmp_s_vec = np.dot(np.ones((1, len(s)), dtype=np.float32), tmp)
            norm = np.linalg.norm(tmp_s_vec, axis=1, keepdims=True)
            res_array.append((tmp_s_vec/norm)[0])

        logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        sum = 0

        for i in range(0, 10):
            train_X, test_X, train_y, test_y = train_test_split(pd.DataFrame(res_array).fillna(0), run_type, test_size=0.2)
            logistic_model.fit(train_X, train_y)
            score = logistic_model.score(test_X, test_y)
            print("TEST", i, "Accuracy：", score)
            sum += score
        print("Logistic Regression Accuracy：", sum / 10)
        print(test_y[:10])
        print(logistic_model.predict(test_X)[:10])

        # return res_array


if __name__ == "__main__":
    print("hello")
    NewPre.word2vec_test()


    '''
    @staticmethod
    def word2vec_model(model_file=r'D:\2021-spring\城市数据挖掘\word2vec-tutorial-master\poi2vec.model'):
        word_model = gensim.models.Word2Vec.load(model_file)
        pretrained_weights = word_model.wv.vectors
        vocab_size, emdedding_size = pretrained_weights.shape

        print('\nPreparing the data for LSTM...')
        sentences, max_sentence_len = NewPre.get_sentense_from_file(NewPre.read_data())
        train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
        train_y = np.zeros([len(sentences)], dtype=np.int32)
        for i, sentence in enumerate(sentences):
            try:
                for t, word in enumerate(sentence[:-2]):
                    train_x[i, t] = NewPre.word2idx(word)
                train_y[i] = NewPre.word2idx(sentence[-2])
            except:
                print(sentence)
        print('train_x shape:', train_x.shape)
        print('train_y shape:', train_y.shape)
        print('\nTraining LSTM...')

        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
        model.add(LSTM(units=emdedding_size))
        model.add(Dense(units=vocab_size))
        model.add(Activation('softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.fit(train_x, train_y, batch_size=128, epochs=100, )
        model.save("./mv_model.h5")
    '''




