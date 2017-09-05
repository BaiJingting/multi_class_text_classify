# -*- coding:utf-8 -*-

import jieba
import cPickle as pickle
import xlrd
from math import sqrt
from collections import defaultdict


def const_Q_A_dict(train_file, filter_words, save=True, save_path=None):
    ind = 1
    answer_ind = {}
    ind_answer = {}
    question_answer = {}
    question_seg = {}

    for file in train_file:
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
            if question not in question_answer:
                question_answer[question] = answer_ind[answer]
                question_seg[question] = set(list(jieba.cut(question)))
                for word in filter_words:
                    if word in question_seg[question]:
                        question_seg[question].remove(word)
        if save:
            with open(save_path + "answer_ind.txt", 'w') as f:
                pickle.dump(answer_ind, f)
            with open(save_path + "ind_answer.txt", 'w') as f:
                pickle.dump(ind_answer, f)
            with open(save_path + "question_answer.txt", 'w') as f:
                pickle.dump(question_answer, f)
            with open(save_path + "question_seg.txt", 'w') as f:
                pickle.dump(question_seg, f)
    dicts = [answer_ind, ind_answer, question_answer, question_seg]
    return dicts


def load_Q_A_dict(load_path):
    with open(load_path + "answer_ind.txt", 'r') as f:
        answer_ind = pickle.load(f)
    with open(load_path + "ind_answer.txt", 'r') as f:
        ind_answer = pickle.load(f)
    with open(load_path + "question_answer.txt", 'r') as f:
        question_answer = pickle.load(f)
    with open(load_path + "question_seg.txt", 'r') as f:
        question_seg = pickle.load(f)
    dicts = [answer_ind, ind_answer, question_answer, question_seg]
    return dicts


def find_answer_ind(nearest_questions, question_answer, k):
    aux = defaultdict(lambda: 0)
    for item in nearest_questions:
        if item[1][1] < 0.15 or item[1][0] < 2:
            continue
        answer = question_answer[item[0]]
        aux[answer] += 1
    # most = 0
    # most_answer = -1
    # for k, v in aux.items():
    #     if v > most:
    #         most = v
    #         most_answer = k
    most_answer = sorted(aux.items(), key=lambda x: x[1], reverse=True)
    k = k if k < len(most_answer) else len(most_answer)
    ret = []
    for i in range(k):
        ret.append(most_answer[i][0])
    return ret


def find_nearest_question(question, question_seg, nearest_K):
    question_dist = {}
    for k, v in question_seg.items():
        if len(v) == 0:
            continue
        l = len(question & v)
        dist = len(question & v) * 1.0 / sqrt((len(question) + 1) * (len(v) + 1))
        question_dist[k] = [l, dist]
    nearest_questions = sorted(question_dist.items(), key=lambda x: x[1][1], reverse=True)[:nearest_K]
    return nearest_questions


def replace_cut(question, filter_words):
    question = question.encode('utf-8').replace('，', '').replace('？', '').replace('。', '') \
        .replace('！', '').replace('…', '')
    question = set(list(jieba.cut(question)))
    ret = set()
    for word in question:
        word = word.encode('utf-8')
        if word not in filter_words:
            ret.add(word)
    return ret


def answer_by_kw_match(question, ind_keywords):
    question = question.encode('utf-8').replace('，', '').replace('？', '').replace('。', '') \
        .replace('！', '').replace('…', '')
    question = set(list(jieba.cut(question)))
    score = [0 for i in range(len(ind_keywords) + 1)]
    for word in question:
        for ind, keywords in ind_keywords.items():
            if word in keywords:
                score[ind] += 1
    label = score.index(max(score))
    return label if label > 0 else -1


def answer_by_KNN(question, dicts, nearest_K, filter_words):
    answer_ind, ind_answer, question_answer, question_seg = dicts[0], dicts[1], dicts[2], dicts[3]
    question = replace_cut(question, filter_words)
    for word in filter_words:
        if word in question:
            question.remove(word)
    if len(question) == 0:
        return None, None
    nearest_questions = find_nearest_question(question, question_seg, nearest_K)
    label = find_answer_ind(nearest_questions, question_answer, 3)
    if len(label) == 0:
        status = 0
        return ["您好，该问题将流转至业务部门修复，请您耐心等待。感谢反馈，祝您生活愉快。".decode('utf-8')], status
    else:
        status = 1
        ret = []
        for ind in label:
            ret.append(ind_answer[ind])
        return ret, status


def load_filter_word(path):
    filter_words = set()
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            filter_words.add(line)
    return filter_words


def extract(init_line):
    line = init_line.split('【客服】')
    content = line[0].split('【用户】')
    text = ""
    for i in range(1, len(content)):
        l = content[i]
        if l.find('>') != -1:
            l = l.split('>', 1)[1]
        if i == len(content) - 1 and len(line) == 1:
            l = l.split(',')
            l = l[0]
        l = l.strip(' ').strip(';').lstrip(' ').strip('\n').strip('\t')
        if l.startswith("https://"):
            continue
        l = l.split("【机器人】")[0]
        text += l
    return text


def find_sim_problems(question, problem_seg, nearest_K, filter_words):
    question = replace_cut(question, filter_words)
    if len(question) == 0:
        return None
    nearest_questions = find_nearest_question(question, problem_seg, nearest_K)
    ret = []
    for item in nearest_questions:
        ret.append(item[0])
    return ret


def extract_(line):
    line = line.split(',', 8)[-1]
    line = line.split(',')
    ret = ""
    for i in range(len(line)-4):
        ret += line[i]
    return ret


def read_filter_words(path):
    filter_words = set()
    with open(path) as f:
        for line in f.readlines():
            filter_words.add(line.strip('\n').strip())
    return filter_words


def find_keywords(words_dict, filter_words):
    keywords = words_dict.keys()
    for word in keywords:
        if word.encode('utf-8') in filter_words:
            keywords.remove(word)
    return keywords
