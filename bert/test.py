# import tensorflow as tf
from bert_serving.client import BertClient
bc = BertClient()
# print(bc.encode(['中国', '美国']))
# b = bc.encode(['中国', '美国'])


if __name__ == "__main__":
    print(bc.encode(["我爱我的祖国"]).shape)

