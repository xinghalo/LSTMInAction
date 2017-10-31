# -*- coding: utf-8 -*-

import sys
import collections
import jieba
import jieba.posseg
import numpy as np
import json
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.layers.pooling import MaxPooling1D
from keras.models import Sequential

reload(sys)  # Python2.7 初始化后会删除 sys.setdefaultencoding 这个方法，我们需要重新载入
sys.setdefaultencoding("utf-8")

MAX_LEN = 20
EMBED_SIZE = 400
HIDDEN_LAYER_SIZE = 256  # 64
NB_CLASSES = 3
BATCH_SIZE = 64
NUM_EPOCHS = 5

STOPWORDS = "/Users/xingoo/PycharmProjects/LSTMInAction/data/stopword.txt"
INTENTTRAIN = "/Users/xingoo/PycharmProjects/LSTMInAction/data/intent_train.txt"

np.random.seed(7)
stopword = [line.strip().decode("utf-8") for line in open(STOPWORDS).readlines()]



def jieba_tokenization(sentence):
    sentence = sentence.strip().lower()

    word_list = []
    ner_list = []
    noun_list = []
    seg = jieba.posseg.cut(sentence)
    for i in seg:
        if "ner" in i.flag:
            word_list.append(i.flag)
            ner_list.append(i)
        elif i.flag == "n":
            noun_list.append(i.word)
        else:
            word_list.append(i.word)

    word_list = filter(lambda x: x and x != " ", word_list)  # filtered empty str

    tokens = " ".join(word_list)

    return [tokens.strip(), word_list, ner_list, noun_list]

def sent2vec(sent):
    xs = []
    words = [x.lower() for x in jieba_tokenization(sent)[1]]
    wids = []
    for word in words:
        wid = word2index[word]
        wids.append(wid)
    xs.append(wids)

    return pad_sequences(xs, maxlen=MAX_LEN)



counter = collections.Counter()
f = open(INTENTTRAIN, "rb")
maxlen = 0
for line in f:
    try:
        _, sent = line.strip().split("	")
        sent = sent.decode("utf-8")
        words = [x.lower() for x in jieba_tokenization(sent)[1]]
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            counter[word] += 1
    except:
        pass

f.close()

word2index = collections.defaultdict(int)
# 获取最常见的5000个词，wid就是一个序号而已
for wid, word in enumerate(counter.most_common(5000)):
    word2index[word[0]] = wid + 1
vocab_sz = len(word2index) + 1
index2word = {v: k for k, v in word2index.items()}


xs, ys = [], []
f = open(INTENTTRAIN, "rb")
for line in f:
    try:
        label, sent = line.strip().split("	")
        sent = sent.decode("utf-8")
        ys.append(int(label))
        words = [x.lower() for x in jieba_tokenization(sent)[1]]
        wids = [word2index[word] for word in words]
        xs.append(wids)
    except:
        pass
f.close()
X = pad_sequences(xs, maxlen=MAX_LEN)
Y = np_utils.to_categorical(ys)

# Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=RANDOM_SEED)
Xtrain, Ytrain = X, Y

# LSTM network
model = Sequential()
model.add(Embedding(vocab_sz, EMBED_SIZE, input_length=MAX_LEN))
model.add(MaxPooling1D(pool_size=2, strides=1))
model.add(SpatialDropout1D(Dropout(0.2)))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(NB_CLASSES))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
# history = model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(Xtest, Ytest))
history = model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

xs = sent2vec(u'我想买男装')
predictions = model.predict(xs)

Xpred = map(lambda x: np.argmax(x), predictions)
print json.dumps(map(lambda x: [index2word.get(k, "") for k in x], xs), ensure_ascii=False)
# print map(lambda x: list(x).index(max(list(x))), predictions)
print predictions

predicted = map(lambda x: {0: "chat", 1: "product", 2: "qa"}[x], Xpred)

print predicted