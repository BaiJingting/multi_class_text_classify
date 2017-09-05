# -*- coding:utf-8 -*-

import os
import jieba.analyse
import xlrd
from gensim import corpora
from sklearn.naive_bayes import MultinomialNB

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file = ROOT_PATH + "/data/init_data/地图用户反馈答复语料20170825.xlsx"
filter_words_path = ROOT_PATH + "/data/user_dict/filter_words.txt"
jieba.load_userdict(ROOT_PATH + "/data/user_dict/user_dict.txt")

data = xlrd.open_workbook(file)
table = data.sheets()[0]
ncol = table.ncols
nrow = table.nrows
q_ind = -1
a_ind = -1
for i in range(ncol):
    if table.row(0)[i].value.encode('utf-8') == 'Q':
        q_ind = i
    if table.row(0)[i].value.encode('utf-8') == 'A':
        a_ind = i
if q_ind == -1 or a_ind == -1:
    print "没找到Q_A两列或文件中列名不为Q_A"
    exit(-1)

label = []
question_seg = []
answer_ind = {}
ind_answer = {}

ind = 1

for i in range(1, nrow):
    question = table.row(i)[q_ind].value
    answer = table.row(i)[a_ind].value
    if answer not in answer_ind:
        answer_ind[answer] = ind
        ind_answer[ind] = answer
        ind += 1
    question = question.encode('utf-8').replace('，', '').replace('？', '').replace('。', '')\
        .replace('！', '').replace('…', '')
    question = list(jieba.cut(question))
    question_seg.append(question)
    label.append(answer_ind[answer])

dictionary = corpora.Dictionary(question_seg)
dictionary.save(ROOT_PATH + '/data/mid_data/dictionary_Naive_Bayes.dict')

length = len(dictionary.token2id)
print length

train_x = []
for i in range(len(question_seg)):
    vec = dictionary.doc2bow(question_seg[i])
    new_vec = [0 for n in range(length)]
    for item in vec:
        new_vec[item[0]] = item[1]
    train_x.append(new_vec)


model = MultinomialNB()
model.fit(train_x, label)


# 预测
data_ = xlrd.open_workbook(ROOT_PATH + "/data/init_data/评估样本-地图.xlsx")
tables = data_.sheets()

with open(ROOT_PATH + "/data_predict/predict_Naive_Bayes.txt", 'w') as f:
    for table in tables:
        ncol = table.ncols
        for i in range(ncol):
            if table.row(0)[i].value.encode('utf-8') == "反馈内容":
                q_ind = i
            if table.row(0)[i].value.encode('utf-8') == "机器人回复内容":
                a_ind = i
        nrow = table.nrows

        for i in range(1, nrow):
            question = table.row(i)[q_ind].value
            question = unicode(question) if not isinstance(question, unicode) else question
            question = question.strip('\n')
            vec = question.encode('utf-8').replace('，', '').replace('？', '').replace('。', '') \
                .replace('！', '').replace('…', '')
            vec = list(jieba.cut(vec))
            vec = dictionary.doc2bow(vec)
            new_vec = [0 for n in range(length)]
            for item in vec:
                new_vec[item[0]] = item[1]
            pred = model.predict(new_vec)
            f.write(question.encode('utf-8'))
            f.write('\t')
            f.write(ind_answer[pred[0]].encode('utf-8'))
            f.write('\n')

