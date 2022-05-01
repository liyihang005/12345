from worker import *
from tqdm import tqdm
from bert_serving.client import BertClient
bc = BertClient()
# print(bc.encode(['中国', '美国']))


def word2vec_lr_test():
    raw_data, r_data, cp1, cp2, cp3 = NewPre.read_data()
    tp_1_name_list = NewPre.get_tp_star_name_list(cp1)
    tp_2_name_list = NewPre.get_tp_star_name_list(cp2)
    tp_3_name_list = NewPre.get_tp_star_name_list(cp3)
    r_data_3, needed_cmps_tp3 = NewPre.get_main_complaints(raw_data, tp_3_name_list, 60, 3)
    r_data_2, needed_cmps_tp2 = NewPre.get_main_complaints(raw_data, tp_2_name_list, 60, 2)
    r_data_1, needed_cmps_tp1 = NewPre.get_main_complaints(raw_data, tp_1_name_list, 60, 1)

    #Type3
    # run_data = r_data_3
    # run_type = needed_cmps_tp3

    # Type2
    # run_data = r_data_2
    # run_type = needed_cmps_tp2

    #type1
    run_data = r_data_1
    run_type = needed_cmps_tp1

    sentences, _ = NewPre.get_sentense_from_file(run_data)
    # model = gensim.models.word2vec.Word2Vec(min_count=1)
    # model.build_vocab(sentences)
    # model.train(sentences, total_examples=model.corpus_count, epochs=100)
    res_array = []
    for s in tqdm(sentences):
        res_array.append(bc.encode([''.join(s)])[0])

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


if __name__ == "__main__":
    print("hello")
    word2vec_lr_test()