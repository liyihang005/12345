from worker import *
from math import sqrt
from keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
import gc
from copy import deepcopy
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE


def correct_train_data(np_data):
    where_nan = np.isnan(np_data)
    where_inf = np.isinf(np_data)
    np_data[where_nan] = 0
    np_data[where_inf] = 0

    return np_data


def correct_type_y(nums):
    # ori_nums = deepcopy(nums)
    result = []
    # for i in np.argsort(nums):
    #     result.append([i])
    tmp = {}
    i = 0
    for num in nums:
        if num not in tmp.keys():
            tmp[num] = i
            i += 1
    for num in nums:
        result.append([tmp[num]])


    # set_nums = list(set(nums))
    # i = 0
    # for setget in set_nums:
    #     nums[nums == setget] = i
    #     i += 1
    return np.array(result)


def word2vec_lr_test():
    raw_data, r_data, cp1, cp2, cp3 = NewPre.read_data()
    tp_1_name_list = NewPre.get_tp_star_name_list(cp1)
    tp_2_name_list = NewPre.get_tp_star_name_list(cp2)
    tp_3_name_list = NewPre.get_tp_star_name_list(cp3)


    r_data_3, needed_cmps_tp3 = NewPre.get_main_complaints(raw_data, tp_3_name_list, 60, 3)
    r_data_2, needed_cmps_tp2 = NewPre.get_main_complaints(raw_data, tp_2_name_list, 60, 2)
    r_data_1, needed_cmps_tp1 = NewPre.get_main_complaints(raw_data, tp_1_name_list, 60, 1)

    # run_data = r_data_3
    # run_type = needed_cmps_tp3

    # Type2
    # run_data = r_data_2
    # out_run_type = needed_cmps_tp2
    # run_type = needed_cmps_tp2

    # Type1
    run_data = r_data_1
    out_run_type = needed_cmps_tp1
    run_type = needed_cmps_tp1

    sentences, _ = NewPre.get_sentense_from_file(run_data)
    model = gensim.models.word2vec.Word2Vec(min_count=1)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=10)
    res_array = []
    for s in sentences:
        tmp = np.zeros((len(s), len(model.wv[sentences[0][0]])), dtype=np.float32)
        for w_i in range(len(s)):
            tmp[w_i] = model.wv[s[w_i]]
        tmp_s_vec = np.dot(np.ones((1, len(s)), dtype=np.float32), tmp)
        norm = np.linalg.norm(tmp_s_vec, axis=1, keepdims=True)
        res_array.append((tmp_s_vec / norm)[0])

    pretrained_weights = model.wv.vectors
    vocab_size, emdedding_size = pretrained_weights.shape
    # train_x = pd.DataFrame(res_array).fillna(0)
    # train_y = pd.DataFrame(run_type)


    sum = 0
    tmp = correct_type_y(run_type)
    run_type = to_categorical(tmp)
    run_type = np.array(run_type)
    res_array = correct_train_data(np.array(res_array))

    train_X = res_array#.reshape((res_array.shape[0], 1, res_array.shape[1]))
    train_y = run_type
    #----------------------------------------------------------------------------------------
    x_train, y_train = train_X, train_y
    # ??????????????????????????????
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=100))  # 64??????????????????????????????????????????100
    model.add(Dropout(0.5))  # ???????????????
    model.add(Dense(64, activation='relu'))  # ?????????
    model.add(Dropout(0.5))  # ???????????????
    model.add(Dense(len(set(out_run_type)), activation='softmax'))

    model.summary()

    # ??????
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # ????????????
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    over_samples = SMOTE(random_state=111)
    x_train, y_train = over_samples.fit_resample(x_train, y_train)

    # ??????
    model.fit(x_train, y_train, epochs=100, batch_size=128)

    # ??????
    score = model.evaluate(x_train, y_train, batch_size=128)
    print(score)
    # ----------------------------------------------------------------------------------------

    # for i in range(0, 10):
    #     # train_X, test_X, train_y, test_y = train_test_split(pd.DataFrame(res_array.reshape((res_array.shape[0], 1, res_array.shape[1]))).fillna(0), run_type.reshape((run_type.shape[0], 1, run_type.shape[1])), test_size=0.2)
    #     print('train_x shape:', train_X.shape)
    #     print('train_y shape:', train_y.shape)
    #     print('\nTraining LSTM...')
    #     model_pre = Sequential()
    #     # model_pre.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
    #     model_pre.add(LSTM(5, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    #     # model_pre.add(LSTM(50, activation='relu', input_shape=(train_X.shape[0], train_X.shape[1])))
    #     # model_pre.add(Dense(1))
    #     model_pre.add(Dense(1, Activation('softmax')))
    #     # model_pre.add(Activation('softmax'))
    #     model_pre.summary()
    #     model_pre.compile(optimizer='adam', loss='mae', metrics=['accuracy'])#sparse_categorical_crossentropy
    #     model_pre.fit(train_X, train_y)
    #     # score = model_pre.score(test_X, test_y)
    #     score = np.average(np.array(train_y) - np.array(model_pre.predict(train_X)))
    #     print("TEST", i, "Accuracy???", score)
    #     sum += score
    # print("Logistic Regression Accuracy???", sum / 10)
    # print(train_y[:10])
    # print(model_pre.predict(train_X)[:10])
    # print(test_y[:10])
    # print(model_pre.predict(test_X)[:10])


if __name__ == "__main__":
    gc.collect()
    print("hello")
    word2vec_lr_test()
    # print(correct_type_y([122,122,22, 33,55,33]))
