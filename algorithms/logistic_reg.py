from worker import *
from smote.main import *
from imblearn.over_sampling import SMOTE

def word2vec_lr_test():
    raw_data, r_data, cp1, cp2, cp3 = NewPre.read_data()
    tp_1_name_list = NewPre.get_tp_star_name_list(cp1)
    tp_2_name_list = NewPre.get_tp_star_name_list(cp2)
    tp_3_name_list = NewPre.get_tp_star_name_list(cp3)
    r_data_3, needed_cmps_tp3 = NewPre.get_main_complaints(raw_data, tp_3_name_list, 60, 3)
    r_data_2, needed_cmps_tp2 = NewPre.get_main_complaints(raw_data, tp_2_name_list, 20, 2)
    r_data_1, needed_cmps_tp1 = NewPre.get_main_complaints(raw_data, tp_1_name_list, 5, 1)

    #Type3
    run_data = r_data_3
    run_type = needed_cmps_tp3

    # Type2
    # run_data = r_data_2
    # run_type = needed_cmps_tp2

    # #type1
    # run_data = r_data_1
    # run_type = needed_cmps_tp1

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
        res_array.append((tmp_s_vec / norm)[0])

    logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    sum = 0
    over_samples = SMOTE(random_state=111)

    for i in range(0, 10):
        train_X, test_X, train_y, test_y = train_test_split(pd.DataFrame(res_array).fillna(0), run_type, test_size=0.2)

        train_X, train_y = over_samples.fit_resample(train_X, train_y)

        logistic_model.fit(train_X, train_y)
        train_score = logistic_model.score(train_X, train_y)
        score = logistic_model.score(test_X, test_y)
        print("TRAIN", i, 'Accuracy: ', train_score)
        print("TEST", i, "Accuracy：", score)
        sum += score
    print("Logistic Regression Accuracy：", sum / 10)
    print(test_y[:10])
    print(logistic_model.predict(test_X)[:10])


if __name__ == "__main__":
    print("hello")
    word2vec_lr_test()