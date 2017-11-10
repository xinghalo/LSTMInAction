# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import jieba
import jieba.posseg as pseg

jieba.load_userdict("/Users/xingoo/PycharmProjects/LSTMInAction/data/pos.txt")

# words = pseg.cut("我最喜欢的牌子是彪马")
words = pseg.cut("我是邢海龙")
for word, flag in words:
    print('%s %s' % (word, flag))