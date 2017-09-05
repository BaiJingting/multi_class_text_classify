# -*- coding:utf-8 -*-

import os
import sys
from collections import defaultdict
import jieba
import xlrd

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

import aux_functions as aux

nearest_K = 5
jieba.load_userdict(ROOT_PATH + "/data/user_dict/user_dict.txt")
jieba.suggest_freq(('不能', '自己'), True)
jieba.suggest_freq(('有', '限'), True)
filter_words = aux.load_filter_word(ROOT_PATH + "/data/user_dict/filter_words.txt")

file = "/Users/baijingting/Downloads/data/百度地图_百度地图_iOS_90109.csv"

problem_line = defaultdict(lambda: "")
problem_seg = defaultdict(lambda: [])

with open(file, 'r') as f:
    line = f.readline()
    for line in f.readlines():
        problem = aux.extract(line)
        if problem not in problem_line:
            problem_line[problem] = aux.extract_(line)
            tmp = set(list(jieba.cut(problem)))
            s = set()
            for word in tmp:
                word = word.encode('utf-8')
                if word not in filter_words:
                    s.add(word)
            problem_seg[problem] = s

data_ = xlrd.open_workbook(ROOT_PATH + "/data/init_data/评估样本-地图.xlsx")
tables = data_.sheets()

with open(ROOT_PATH + "/data_predict/sim_problems.txt", 'w') as f:
    for table in tables[:1]:
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
            sim_problems = aux.find_sim_problems(question, problem_seg, nearest_K, filter_words)
            if sim_problems is None:
                continue
            f.write("======================================================\n")
            f.write(question.encode('utf-8'))
            f.write('\n')
            f.write('-------------------------------------------\n')
            for t in sim_problems:
                f.write(problem_line[t])
                f.write('\n')
            f.write('\n')

