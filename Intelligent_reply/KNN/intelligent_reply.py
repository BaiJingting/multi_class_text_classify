# -*- coding:utf-8 -*-

import os
import sys
import jieba
import xlrd

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

import aux_functions as aux

nearest_K = 15

jieba.load_userdict(ROOT_PATH + "/data/user_dict/user_dict.txt")
filter_words = aux.load_filter_word(ROOT_PATH + "/data/user_dict/filter_words.txt")

# # 构造问答字典并保存
# file1 = ROOT_PATH + "/data/init_data/训练语料-地图.xlsx"
# file2 = ROOT_PATH + "/data/init_data/地图用户反馈答复语料20170825.xlsx"
# train_file = [file1, file2]
# dicts = aux.const_Q_A_dict(train_file, filter_words, save=True, save_path=ROOT_PATH + "/data/mid_data/")

dicts = aux.load_Q_A_dict(load_path=ROOT_PATH + "/data/mid_data/")

data_ = xlrd.open_workbook(ROOT_PATH + "/data/init_data/评估样本-地图.xlsx")
tables = data_.sheets()

with open(ROOT_PATH + "/data_predict/predict_KNN.txt", 'w') as f:
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

            answer, status = aux.answer_by_KNN(question, dicts, nearest_K, filter_words)
            if status is None:
                continue
            f.write(question.encode('utf-8'))
            for t in answer:
                f.write('\t')
                f.write(t.encode('utf-8'))
            f.write('\t')
            f.write(str(status))
            f.write('\n')
