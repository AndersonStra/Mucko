
import json
import os
import json
import itertools
import pickle
import re
from collections import Counter
import yaml
from tqdm import tqdm
from vocabulary import Vocabulary
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np


punctuation = '!,;:?"\''


def removePunctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation), '', text)
    return text.strip().lower()

def extract_word_count():
    questions = []

    with open('/home/data1/yjgroup/hdd/okvqa/question_raw_train.json', 'r') as f:
        qa_raw = json.load(f)

    with open('/home/data1/yjgroup/hdd/okvqa/question_raw_val.json', 'r') as f:
        qa_raw_val = json.load(f)
        qa_raw.update(qa_raw_val)


    for qid, item in qa_raw.items():
        question = item['question'].lower()
        question = question.strip()
        question = removePunctuation(question)
        
        questions.append(question.split(' '))
        


    all_tokens = itertools.chain.from_iterable(questions)
    counter = Counter(all_tokens)

    word_count = dict(counter)

    with open('/home/data1/yjgroup/hdd/okvqa/word_count2.json', 'w') as f:
        json.dump(word_count, f)
    print('finished!!!')

def process_q():
    with open('/home/data1/yjgroup/hdd/okvqa/question_raw_train.json', 'r') as f:
        qa_raw = json.load(f)


    with open('/home/data1/yjgroup/hdd/okvqa/question_raw_val.json', 'r') as f:
        qa_raw_val = json.load(f)

    qa_raw_dict={}
    for qid, item in qa_raw.items():
        question = item['question'].lower()
        question = question.strip()
        question = removePunctuation(question)
        item['question'] = question
        qa_raw_dict[qid] = item
    with open('/home/data1/yjgroup/hdd/okvqa/question_raw_train6.json', 'w') as f:
        json.dump(qa_raw_dict, f)
    
    qa_raw_val_dict = {}
    for qid, item in qa_raw_val.items():
        question = item['question'].l ower()
        question = question.strip()
        question = removePunctuation(question)
        item['question'] = question
        qa_raw_val_dict[qid] = item

    with open('/home/data1/yjgroup/hdd/okvqa/question_raw_val6.json', 'w') as f:
        json.dump(qa_raw_val_dict, f)
    print('finish')
    
        
    
# def get_glove_embed():

#     glovevocabulary = Vocabulary(
#         '/home/data1/yjgroup/hdd/okvqa/word_count2.json', min_count=1)

#     # 读取glove模型
#     glove_300d_word2vec_file = '/home/data1/yjgroup/hdd/pr_okvqa_memory/glove.6B.300d.w2v.txt'
#     print('loading glove model...')
#     glove_model = KeyedVectors.load_word2vec_format(glove_300d_word2vec_file, binary=False)
#     print('finished load glove model...')

#     embeds = []

#     w2i = glovevocabulary.word2index
#     for word in w2i.keys():
#         try:
#             embed = glove_model[word]
#         except KeyError:
#             embed = np.zeros(300, dtype=np.float32)
#         embeds.append(embed)
#     embeds = np.stack(embeds)
#     np.save('/home/data1/yjgroup/hdd/okvqa/glove300dvocab2.npy', embeds)
#     print('finish')

def get_glove_embed():

    glovevocabulary = Vocabulary(
        '/home/data1/yjgroup/hdd/okvqa/word_count2.json', min_count=1)

    # 读取glove模型
    glove_300d_word2vec_file = '/home/data1/yjgroup/hdd/pr_okvqa_memory/glove.6B.300d.w2v.txt'
    print('loading glove model...')
    glove_model = KeyedVectors.load_word2vec_format(glove_300d_word2vec_file, binary=False)
    print('finished load glove model...')

    embeds = []

    w2i = glovevocabulary.word2index
    for word in w2i.keys():
        try:
            embed = glove_model[word]
        except KeyError:
            embed = np.zeros(300, dtype=np.float32)
        embeds.append(embed)
    embeds = np.stack(embeds)
    np.save('/home/data1/yjgroup/hdd/okvqa/glove300dvocab2.npy', embeds)
    print('finish')


if __name__ == "__main__":
    # extract_word_count()
    # get_glove_embed()
    process_q()
