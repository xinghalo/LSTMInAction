# -*- coding: utf-8 -*-
import sys
reload(sys)  # Python2.7 初始化后会删除 sys.setdefaultencoding 这个方法，我们需要重新载入
sys.setdefaultencoding("utf-8")

import jieba.posseg

seg = jieba.posseg.cut("我最喜欢的就是白色的外套")
for i in seg:
    print i.word,i.flag

print "--------------------"

seg = jieba.posseg.cut("谢霆锋的爸爸是谢贤")
for i in seg:
    print i.word,i.flag