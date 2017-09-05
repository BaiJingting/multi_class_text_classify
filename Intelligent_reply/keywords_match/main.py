# -*- coding:utf-8 -*-

import os
import sys
from collections import defaultdict
import jieba.analyse
import xlrd

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

import aux_functions as aux

file = ROOT_PATH + "/data/init_data/地图用户反馈答复语料20170825.xlsx"
filter_words_path = ROOT_PATH + "/data/user_dict/filter_words.txt"
jieba.load_userdict(ROOT_PATH + "/data/user_dict/user_dict.txt")

ind = 1
answer_ind = {}
ind_answer = {}

answer_question = defaultdict(lambda: defaultdict(lambda: 0))
answer_num = defaultdict(lambda: 0)

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

for i in range(1, nrow):
    question = table.row(i)[q_ind].value
    answer = table.row(i)[a_ind].value
    if answer not in answer_ind:
        answer_ind[answer] = ind
        ind_answer[ind] = answer
        ind += 1
    answer_num[answer_ind[answer]] += 1
    question = question.encode('utf-8').replace('，', '').replace('？', '').replace('。', '') \
        .replace('！', '').replace('…', '')
    question = set(list(jieba.cut(question)))
    for word in question:
        answer_question[answer_ind[answer]][word] += 1

ind_keywords = {}

filter_words = aux.read_filter_words(filter_words_path)

for ind, words_dict in answer_question.items():
    keywords = aux.find_keywords(words_dict, filter_words)
    print ind
    print " ".join(keywords)
    ind_keywords[ind] = keywords

data_ = xlrd.open_workbook(ROOT_PATH + "/data/init_data/评估样本-地图.xlsx")
tables = data_.sheets()

for table in tables:
    ncol = table.ncols
    for i in range(ncol):
        if table.row(0)[i].value.encode('utf-8') == "反馈内容":
            q_ind = i
        if table.row(0)[i].value.encode('utf-8') == "机器人回复内容":
            a_ind = i
    nrow = table.nrows

    with open(ROOT_PATH + "/data_predict/predict_keywords_match.txt", 'w') as f:
        for i in range(1, nrow):
            question = table.row(i)[q_ind].value
            question = unicode(question) if not isinstance(question, unicode) else question
            question = question.strip('\n')
            label = aux.answer_by_kw_match(question, ind_keywords)
            if label == -1:
                answer = "您好，该问题将流转至业务部门修复，请您耐心等待。感谢反馈，祝您生活愉快。".decode('utf-8')
            else:
                answer = ind_answer[label]
            f.write(question.encode('utf-8'))
            f.write('\t')
            f.write(answer.encode('utf-8'))
            f.write('\n')
