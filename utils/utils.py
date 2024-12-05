import torch
from utils.bleu import bleuFromMaps
from utils.cal_rouge import calculate_rouge_hf, calculate_meteor_hf
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from tree_sitter import Language, Parser
# from vllm import LLM, SamplingParams
from nltk.stem import WordNetLemmatizer
import json
import os
from strsimpy.levenshtein import Levenshtein
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
import random
from datetime import datetime
lemmatizer = WordNetLemmatizer()

IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import statistics",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "go": [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
    ],
    "cpp": [
        "using namespace std;",      
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<cmath>",
        "#include<math.h>",
        "#include<numeric>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<set>",
        "#include<map>",
        "#include<queue>",
        "#include<stack>",
        "#include<list>",
        "#include<deque>",
        "#include<boost/any.hpp>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
        "#include<sstream>",
        "#include<fstream>",
    ],
}


python_dict = [('data', 14119), ('name', 13844), ('value', 9501), ('result', 8697), ('cls', 8038), ('path', 7955), ('key', 6608), \
        ('url', 6548), ('response', 6343), ('args', 6062), ('msg', 4970), ('filename', 4933), ('params', 4810), ('x', 4638), ('request', 4637), ('ret', 4197), \
            ('obj', 3872), ('res', 3413), ('message', 3359), ('f', 3320), ('s', 3285), ('start', 3218), ('r', 3181), ('text', 3137), ('cmd', 2977), \
                ('config', 2940), ('func', 2922), ('n', 2863), ('y', 2797), ('index', 2790), ('context', 2752), ('d', 2697), ('node', 2628), ('p', 2569), ('query', 2436), \
                    ('line', 2432), ('timeout', 2385), ('out', 2332), ('output', 2231), ('user', 2230), ('headers', 2209), ('size', 2115), ('method', 2101), ('i', 2098), \
                        ('kwargs', 2069), ('m', 2056), ('a', 2052), ('values', 1994), ('b', 1951), ('results', 1909), ('val', 1900), ('c', 1874), ('event', 1862), ('model', 1853), \
                            ('content', 1815), ('t', 1808), ('target', 1767), ('lines', 1766), ('options', 1762), ('status', 1739), ('count', 1707), ('parser', 1697), ('source', 1694), \
                                ('end', 1665), ('body', 1660), ('prefix', 1657), ('version', 1647), ('offset', 1630), ('v', 1612), ('match', 1606), ('command', 1602), ('payload', 1597), \
                                    ('item', 1574), ('state', 1559), ('ctx', 1552), ('get', 1543), ('session', 1531), ('client', 1513)]


def add_junk_code(code):
    assignment_num = random.randint(0, 8)
    name_choose = random.sample(python_dict, assignment_num)
    new_sentence = []
    for i in range(assignment_num):
        assignment_sentence = name_choose[i][0] + ' = ' + str(random.randint(0, 100))
        new_sentence.append('    ' + assignment_sentence)

    add_len = 0
    split_idx = 0
    if code.rfind('"""\n') != -1:
        split_idx = code.rfind('"""\n')
        add_len = len('"""\n')
    elif code.rfind("'''\n") != -1:
        split_idx = code.rfind("'''\n")
        add_len = len("'''\n")
    elif code.find(':\n') != -1:
        split_idx = code.find(':\n')
        add_len = len(':\n')
    elif code.find(') {\n') != -1:
        split_idx = code.find(') {\n')
        add_len = len(') {\n')
    split_idx = split_idx + add_len
    new_code = code[:split_idx] + '\n'.join(new_sentence) + '\n' + code[split_idx:]
    return new_code

def remove_format_symbols(code):

    new_code = ""
    for i in range(len(code)):
        if code[i] == ' ':
            k = random.random()
            if k > 0.9:
                new_code += ''
            else:
                new_code += code[i]

        elif code[i] == '\n' or code[i] == ':' or code[i] == '(' or code[i] == ')' or code[i] == '{' or code[i] == '}' or code[i] == '[' or code[i] == ']' or code[i] == ',' or code[i] == '.' or code[i] == '*' or code[i] == '**':
            k = random.random()
            if k > 0.75:
                new_code += ''
            else:
                new_code +=  code[i]

        else:
            new_code += code[i]
    return new_code

def switch_ide(original_ide, flag, last_symbols = None):
    new_dict = dict()
    cc = 0
    for ide in original_ide:
        if flag == 'random':
            tmp = "".join(random.choice(["I", "l"]) for i in range(random.randint(8, 8)))
        elif flag == 'all_random':
            def random_var_name(count):
                ALPHA_NUMERIC_STRING = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                builder = []
                while count != 0:
                    character = random.randint(0, len(ALPHA_NUMERIC_STRING) - 1)
                    builder.append(ALPHA_NUMERIC_STRING[character])
                    count -= 1
                return ''.join(builder)
            tmp = random_var_name(10)
        elif flag == 'mask':
            tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
            if last_symbols != None:
                new_tokens_len = len(tokenizer.tokenize(last_symbols[cc] + ide)) - 1
            else:
                new_tokens_len = len(tokenizer.tokenize(ide))
            tmp = ''.join('<s>' for i in range(new_tokens_len)) + 'B'
        else:
            tmp = "v" + str(cc)
        cc += 1
        new_dict.update({ide: tmp})
    return new_dict

def switch_single_ide(ide, flag):
    if flag == 'mask':
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
        new_tokens_len = len(tokenizer.tokenize(ide)) - 1
        tmp = ''.join('<s>' for i in range(new_tokens_len)) + 'B'
    return tmp
def find_python_ide(code_string): 
    LANGUAGE = Language('build/my-languages.so', "python")
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes(code_string, 'utf-8'))
    root_node = tree.root_node
    nodes = [root_node]

    ilist = []
    new_code = code_string.split('\n')

    while len(nodes) > 0:
        cur_node = nodes[0]
        nodes = nodes[1:]
        if cur_node.type == "function_definition":
            for child in cur_node.children:

                if child.type == "identifier":
                    s = child.start_point
                    e = child.end_point
                    ide = new_code[s[0]][s[-1]:e[-1]]
                    if not ide in ilist: ilist.append(ide)

                elif child.type == "parameters":
                    for gchild in child.children:
                        # print(gchild.type)
                        if gchild.type == "identifier":
                            s = gchild.start_point
                            e = gchild.end_point
                            
                            ide = new_code[s[0]][s[-1]:e[-1]]
                            if not ide in ilist: ilist.append(ide)
                        elif gchild.type == "default_parameter":
                            for ggchild in gchild.children:
                                 if ggchild.type == "identifier":
                                    s = ggchild.start_point
                                    e = ggchild.end_point
                                    
                                    ide = new_code[s[0]][s[-1]:e[-1]]
                                    if not ide in ilist: ilist.append(ide)
                        elif gchild.type == "typed_parameter":
                            for ggchild in gchild.children:
                                 if ggchild.type == "identifier":
                                    s = ggchild.start_point
                                    e = ggchild.end_point
                                    
                                    ide = new_code[s[0]][s[-1]:e[-1]]
                                    if not ide in ilist: ilist.append(ide)
                else:       
                    nodes.append(child)

        elif cur_node.type == "assignment":
            
            for child in cur_node.children:
                if child.type == "identifier":
                    s = child.start_point
                    e = child.end_point

                    ide = new_code[s[0]][s[-1]:e[-1]]
                    if not ide in ilist: ilist.append(ide)

                elif child.type == "pattern_list":
                    for gchild in child.children:
                        if gchild.type == "identifier":
                            s = gchild.start_point
                            e = gchild.end_point
                            
                            ide = new_code[s[0]][s[-1]:e[-1]]
                            if not ide in ilist: ilist.append(ide)
                
                elif child.type == "=":
                    break
        elif cur_node.type == "pattern_list" or cur_node.type == 'for_statement':
            for child in cur_node.children:
                if child.type == "identifier":
                    s = child.start_point
                    e = child.end_point
                    ide = new_code[s[0]][s[-1]:e[-1]]
                    if not ide in ilist: ilist.append(ide)

        else :
            
            for child in cur_node.children:
                nodes.append(child)
    return ilist


def my_find_python_ide(code_string): 
    LANGUAGE = Language('build/my-languages.so', "python")
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes(code_string, 'utf-8'))
    root_node = tree.root_node
    nodes = [root_node]

    ilist = []
    new_code = code_string.split('\n')

    while len(nodes) > 0:
        cur_node = nodes[0]
        nodes = nodes[1:]
        if cur_node.type == "call" or cur_node.type == "type" or cur_node.type == "import_from_statement" or cur_node.type == "import_statement":
            continue
        elif cur_node.type == 'identifier':
            s = cur_node.start_point
            e = cur_node.end_point
            ide = new_code[s[0]][s[-1]:e[-1]]
            if not ide in ilist: ilist.append(ide)

        else :
            for child in cur_node.children:
                nodes.append(child)
    return ilist

def my_find_java_ide(code_string, last_symbol = False): 
    LANGUAGE = Language('build/my-languages.so', "java")
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes(code_string, 'utf-8'))
    root_node = tree.root_node
    nodes = [root_node]

    ilist = []
    last_symbols = []
    new_code = code_string.split('\n')

    while len(nodes) > 0:
        cur_node = nodes[0]
        nodes = nodes[1:]
        if cur_node.type == "method_invocation" or cur_node.type == "import_declaration":
            continue
        elif cur_node.type == 'identifier' and cur_node.parent.type != 'class_declaration':
            s = cur_node.start_point
            e = cur_node.end_point
            ide = new_code[s[0]][s[-1]:e[-1]]
            if not ide in ilist: 
                ilist.append(ide)
                last_symbols.append(new_code[s[0]][s[-1] - 1])

        else :
            for child in cur_node.children:
                nodes.append(child)
    if last_symbol == True:
        return ilist, last_symbols
    else:
        return ilist

def replace_ide(code_string, ide_dict, language = 'python', split = True): 
    LANGUAGE = Language('build/my-languages.so', language)
    parser = Parser()
    parser.set_language(LANGUAGE)
    cnt = 0
    while True:
        cnt += 1
        tree = parser.parse(bytes(code_string, 'utf-8'))
        flag = 0
        root_node = tree.root_node
        nodes = [root_node]
        
        ilist = []
        new_code = code_string.split('\n')
        org_code = code_string.split('\n')
        is_change = [0] * len(new_code)
        if language == 'python':
            while len(nodes) > 0:
                cur_node = nodes[0]
                nodes = nodes[1:]
                if cur_node.type == "type" or cur_node.type == "import_from_statement" or cur_node.type == "import_statement":
                    continue
                elif cur_node.type == 'identifier':
                    s = cur_node.start_point
                    e = cur_node.end_point
                    ide = org_code[s[0]][s[-1]:e[-1]]
                    if ide in ide_dict and is_change[s[0]] == 0:
                        
                        new_code[s[0]] = org_code[s[0]][:s[-1]] + ide_dict[ide] + org_code[s[0]][e[-1]:]
                        is_change[s[0]] = 1
                        flag = 1

                else :
                    for child in cur_node.children:
                        nodes.append(child)

        elif language == 'java':
            while len(nodes) > 0:
                cur_node = nodes[0]
                nodes = nodes[1:]
                if cur_node.type == "import_declaration":
                    continue
                elif cur_node.type == 'identifier' and cur_node.parent.type != 'class_declaration':
                    s = cur_node.start_point
                    e = cur_node.end_point
                    ide = org_code[s[0]][s[-1]:e[-1]]
                    if ide in ide_dict and is_change[s[0]] == 0 and ide !='s' and ide != 'B':
                        ide_dict[ide] = switch_single_ide(org_code[s[0]][s[-1] - 1] + ide, 'mask')
                        left = s[-1]
                        right = e[-1]
                        if org_code[s[0]][s[-1] - 1] == ' ':
                            left = left - 1
                        new_code[s[0]] = org_code[s[0]][:left] + ide_dict[ide] + org_code[s[0]][right:]
                        is_change[s[0]] = 1
                        flag = 1


                else :
                    for child in cur_node.children:
                        nodes.append(child)
        
        code_string = '\n'.join(new_code)
        if flag == 0:
            break
    if split == False:
        return code_string  
    else:

        add_len = 0
        split_idx = 0
        if code_string.rfind('"""\n') != -1:
            split_idx = code_string.rfind('"""\n')
            add_len = len('"""\n')
        elif code_string.rfind("'''\n") != -1:
            split_idx = code_string.rfind("'''\n")
            add_len = len("'''\n")
        elif code_string.find(':\n') != -1:
            split_idx = code_string.find(':\n')
            add_len = len(':\n')
        elif code_string.find(') {\n') != -1:
            split_idx = code_string.find(') {\n')
            add_len = len(') {\n')
        split_idx = split_idx + add_len

        prompt = code_string[:split_idx]
        code = code_string[split_idx:]
        return prompt, code


def decode_from_emb(input_ids, x_emb, y_emb, tokenizer):
    vector_map = {}
    final_ids = torch.zeros(input_ids.shape, dtype=torch.long).to(input_ids.device)
    
    for i in range(len(input_ids)):
        sentence = []
        for j in range(len(input_ids[i])):
            if input_ids[i][j].item() < 5:
                final_ids[i][j] = input_ids[i][j].item()
                sentence.append(input_ids[i][j].item())
            elif input_ids[i][j].item() in vector_map:
                final_ids[i][j] = vector_map[input_ids[i][j].item()]
                sentence.append(vector_map[input_ids[i][j].item()])
            else:
                new_emb = x_emb[i][j].repeat(y_emb.shape[0], 1)
                dist = torch.norm(new_emb - y_emb, dim=1)
                dist[input_ids[i][j]] = 1000000000
                dist[:5] = 1000000000
                dist[32000:] = 1000000000
                idx = torch.argmin(dist)
                vector_map[input_ids[i][j].item()] = idx.item()
                sentence.append(vector_map[input_ids[i][j].item()])
                final_ids[i][j] = idx.item()
        decode_sentence = tokenizer.convert_ids_to_tokens(sentence, skip_special_tokens=True)
    return final_ids

def cal_bleu(ans, ground_truth):
    score = 0
    for i in range(len(ans)):

        score += sentence_bleu([ground_truth[i]], ans[i])
    return score / len(ans)



def set_emb(emb, protect_ids):
    for i in protect_ids:
        emb[i] = 1000000000


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)


    def forward(self, features, **kwargs):

        x = torch.mean(features, dim = 1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    

def gan_loss_fun(logits, is_real):
    if is_real:
        labels = torch.ones_like(logits)
    else:
        labels = torch.zeros_like(logits)
    loss_fct = nn.BCEWithLogitsLoss()
    loss = loss_fct(logits, labels)
    return loss


def nn_project(curr_embeds, embedding_layer, input_ids):

    
    batch_size,seq_len, emb_dim = curr_embeds.shape
    
    # Using the sentence transformers semantic search which is 
    # a dot product exact kNN search between a set of 
    # query vectors and a corpus of vectors

    embedding_matrix = embedding_layer.weight
    # not want to change to /n, fill it with 100000
    # embedding_matrix[13] = torch.tensor([100000] * emb_dim)
    embedding_matrix = normalize_embeddings(embedding_matrix) # corpus
    projected_embeds = torch.zeros_like(curr_embeds)
    nn_indices = torch.zeros_like(input_ids)
    cnt = 0
    for curr_embeds_i in curr_embeds:
        curr_embeds_i = curr_embeds_i.reshape((-1,emb_dim))
        # input_ids = input_ids.reshape((curr_embeds_i.shape[0]))
        curr_embeds_i = normalize_embeddings(curr_embeds_i) # queries

        hits = semantic_search(curr_embeds_i, embedding_matrix, 
                                query_chunk_size=curr_embeds_i.shape[0], 
                                top_k=2,
                                score_function=dot_score)

        for i in range(len(hits)):
            if input_ids[0][i].item() != 13 and hits[i][0]["corpus_id"] == 13:
                hits[i][0]["corpus_id"] = hits[i][1]["corpus_id"]
        nn_indices[cnt] = torch.tensor([hits[i][0]["corpus_id"] for i in range(len(hits))], device=curr_embeds.device)
        projected_embeds[cnt] = embedding_layer(nn_indices[cnt])
        cnt += 1
    return projected_embeds, nn_indices


def new_nn_project(perturb_embedding_layer, embedding_layer, input_ids):

    curr_embeds = perturb_embedding_layer(input_ids) + embedding_layer(input_ids)
    batch_size,seq_len, emb_dim = curr_embeds.shape
    
    # Using the sentence transformers semantic search which is 
    # a dot product exact kNN search between a set of 
    # query vectors and a corpus of vectors
    curr_embeds = curr_embeds.reshape((-1,emb_dim))
    input_ids = input_ids.reshape((curr_embeds.shape[0]))
    curr_embeds = normalize_embeddings(curr_embeds) # queries

    embedding_matrix = embedding_layer.weight
    embedding_matrix = normalize_embeddings(embedding_matrix) # corpus
    
    hits = semantic_search(curr_embeds, embedding_matrix, 
                            query_chunk_size=curr_embeds.shape[0], 
                            top_k=2,
                            score_function=dot_score)

    nn_indices = torch.tensor([hits[i][0]["corpus_id"] for i in range(len(hits))], device=curr_embeds.device)
    projected_embeds = embedding_layer(nn_indices)
    diff = projected_embeds - curr_embeds
    # diff will be updated to the new embedding layer
    # use a mask to set new_perturb_layer = diff + perturb_embedding_layer.weight
    seen = {}
    for i in range(len(nn_indices)):
        if nn_indices[i].item() not in seen:
            seen[nn_indices[i].item()] = 1
            perturb_embedding_layer.weight[i] = diff[i]
    return perturb_embedding_layer.weight, nn_indices.unsqueeze(0)


class GPT2LM:
    def __init__(self, cuda=-1, model_resolution = 'codegpt'):

        import os
        self.lm = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf", torch_dtype=torch.float16)   
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
        self.cuda = cuda
        if self.cuda >= 0:
            self.lm.cuda(self.cuda)

    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """

        ipt = self.tokenizer(sent, return_tensors="pt", max_length=300, verbose=False)

        
        if self.cuda >= 0:
            for k in ipt.keys():
                ipt[k] = ipt[k].cuda(self.cuda)
        ans = self.lm(**ipt, labels=ipt.input_ids)[0]
        return math.exp(ans)


class testSummaryModel():
    def __init__(self, model_type, task = 'Summary'):
        self.llm = LLM(model=model_type, dtype='half', max_model_len = 1024)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.model_type = model_type
        self.task = task
        if task == 'Summary':
            if model_type == 'codellama/CodeLlama-7b-Instruct-hf' or model_type == 'codellama/CodeLlama-13b-Instruct-hf':
                print("use sampling params")
                self.sampling_params = SamplingParams(max_tokens=1000, frequency_penalty = 1,best_of=10, use_beam_search=True,temperature=0)
            elif model_type == 'codellama/CodeLlama-7b-hf':
                self.sampling_params = SamplingParams(max_tokens=1000, frequency_penalty = 1,temperature=0.8)

            with open('data/summary_data/test.jsonl') as f:
                ground_data = f.readlines()
                self.ground_truth = [' '.join(json.loads(line)['docstring_tokens']) for line in ground_data]

        elif task == 'Completion':
            self.sampling_params = SamplingParams(max_tokens=1000, top_p=0.95, temperature=0)
        elif task == 'Refine' or task == 'Translate' or task == 'Translate-CXG':
            self.sampling_params = SamplingParams(max_tokens=1000, top_p=0.95, temperature=0)
        
    def sentence_normalization(self, sent):
        if '.' in sent:
            sent = sent[:sent.find('.')]
        sent = sent + ' .'
        sent_first = sent.split(' ')[0].lower()
        change_tok = lemmatizer.lemmatize(sent_first, pos='v')
        return change_tok + ' ' + ' '.join(sent.split(' ')[1:])
    
    def __call__(self, input_ids_ls, part_num, save_path = None):
        if save_path != None:
            with open(save_path, 'r') as f:
                data_ls = f.readlines()
                for i in range(len(data_ls)):
                    data_ls[i] = json.loads(data_ls[i])
            for i in range(len(input_ids_ls)):
                org_code = self.tokenizer.decode(input_ids_ls[i])
                idx1 = org_code.rfind('<</SYS>>\n\n', -1) + len('<</SYS>>\n\n')
                idx2 = org_code.rfind('\"\"\"Please fill', -1)
                data_ls[i]['new_code'] = org_code

        
        
        # Print the outputs.
        if self.task == 'Summary':
            outputs = self.llm.generate(prompt_token_ids=input_ids_ls, sampling_params=self.sampling_params)

            ground_truth = self.ground_truth[:part_num]
            for i in range(len(outputs)):
                generated_text = outputs[i].outputs[0].text

                if ' The goal of this function is to ' in generated_text:
                    generated_text = generated_text[generated_text.find(' The goal of this function is to ') + len(' The goal of this function is to '):].split(',')[0].split('.')[0].split('\n')[0] + ' .'
                else:
                    generated_text = generated_text.lstrip('\n').split('.')[0].split(',')[0].split('\n')[0].lstrip(' ') + ' .'
                if save_path != None:
                    data_ls[i]['new_output'] = generated_text

                if self.model_type == 'codellama/CodeLlama-7b-Instruct-hf' or "codellama/CodeLlama-13b-Instruct-hf":
                    outputs[i] = [self.sentence_normalization(generated_text)]
                    ground_truth[i] = [self.sentence_normalization(ground_truth[i])]
                elif self.model_type == 'codellama/CodeLlama-7b-hf':
                    outputs[i] = [generated_text]
                    ground_truth[i] = [ground_truth[i]]
            reult, bleu_ls = bleuFromMaps(ground_truth, outputs)
            rouge_score = calculate_rouge_hf(outputs, ground_truth)
            print('ROUGE: ', rouge_score)
            meteor_score = calculate_meteor_hf(outputs, ground_truth)
            print('METEOR: ', meteor_score)
            if save_path != None:
                for i in range(len(bleu_ls)):
                    data_ls[i]['new_bleu'] = bleu_ls[i]
            if save_path != None:
                write_file = open(save_path, 'w')
                for i in range(len(data_ls)):
                    write_file.write(json.dumps(data_ls[i]) + '\n')
                write_file.close()
            return reult[0]
        elif self.task == 'Completion':
            now = datetime.now()


            time_str = now.strftime("%Y%m%d%H%M%S")

            for iii in range(1):
                with open('data/gen_data/HumanEval.jsonl', 'r') as file:
                    data = file.readlines()
                outputs = self.llm.generate(prompt_token_ids=input_ids_ls, sampling_params=self.sampling_params)
                for i in range(len(outputs)):
                    data[i] = json.loads(data[i])
                    generated_text = outputs[i].outputs[0].text
                    data[i]['org_output'] = generated_text

                    left = generated_text.find('```python')
                    if left != -1:
                        left = left + len('```python') + 1
                    else:
                        left = generated_text.find('```')
                        if left != -1:
                            left = left + len('```') + 1

                    right = generated_text.rfind('```')
                    # save to file

                    if left != -1:  
                        data[i]['completion'] = generated_text[left:right].strip()
                    else:
                        data[i]['completion'] = generated_text.strip()
                    data[i]['completion'] = '\n'.join(IMPORT_HELPER['python']) + '\n' + data[i]['completion']

                with open(f'result/HumanEval1_name_{time_str}.jsonl', 'a') as file:
                    for iter in range(len(data)):
                        file.write(json.dumps(data[iter]) + '\n')
            
            # run evaluate_functional_correctness
            os.system(f'python utils/clean_data.py result/HumanEval1_name_{time_str}.jsonl result/HumanEval2_name_{time_str}.jsonl')
            os.system(f'evaluate_functional_correctness result/HumanEval2_name_{time_str}.jsonl complete')
        
        
        elif self.task == 'Translate':
            os.system('rm translation_python.jsonl')
            for k in range(1):
                data = []
                outputs = self.llm.generate(prompt_token_ids=input_ids_ls, sampling_params=self.sampling_params)
                for i in range(len(outputs)):
                    # TODO: 规定标号
                    tmp_data = {}
                    tmp_data['task_id'] = 'Python/' + str(i)
                    tmp_data['prompt'] = ""
                    generated_text = outputs[i].outputs[0].text
                    tmp_data['org_output'] = generated_text
                    left_idx = generated_text.find('```python')
                    if left_idx != -1:
                        right_idx = generated_text.find('```', left_idx + 1)
                        new_code = generated_text[left_idx + len('```python') + 1:right_idx].strip()
                    if left_idx == -1:
                        left_idx = generated_text.find('```')
                        if left_idx != -1:
                            right_idx = generated_text.find('```', left_idx + 1)
                            new_code = generated_text[left_idx + len('```') + 1:right_idx].strip()
                        else:
                            left_idx = generated_text.find('def')
                            right_idx = generated_text.find('\n\n', left_idx + 1)
                            new_code = generated_text[left_idx:right_idx].strip()
                        # save to file
                    if new_code.find('[/PYTHON]') != -1:
                        new_code = new_code[:new_code.find('[/PYTHON]')]
                    if left_idx != -1:  
                        tmp_data['generation'] = new_code
                    else:
                        tmp_data['generation'] = generated_text.strip()
                    def_left = tmp_data['generation'].find('def') + 4
                    def_right = tmp_data['generation'].find('(', def_left + 1)
                    tmp_data['function_name'] = tmp_data['generation'][def_left:def_right].strip()
                    data.append(tmp_data)
                with open('translation_python.jsonl', 'a') as file:
                    for iter in range(len(data)):
                        file.write(json.dumps(data[iter]) + '\n')
                
            
            # run evaluate_functional_correctness
            os.system('bash CodeGeeX/scripts/evaluate_humaneval_x.sh translation_python.jsonl python')

        




    
def ppl_format(input, model_name):
    if model_name == 'codellama/CodeLlama-7b-hf':
        idx1 = input.find('"""')
        idx2 = input.rfind('"""')
        input = input[:idx1] + input[idx2 + 4:]
        return input
    elif model_name == 'codellama/CodeLlama-7b-Instruct-hf':
        idx1 = input.find('<</SYS>>\n\n') + len('<</SYS>>\n\n')
        idx2 = input.find('\n"""Please')
        input = input[idx1:idx2]
        return input
    
class EditDistance:
    def __init__(self, normalized = False):
        self.lev = Levenshtein() if not normalized else NormalizedLevenshtein()
    
    def __call__(self, sentence1, sentence2):
        sentence1, sentence2 = sentence1.lower(), sentence2.lower()
        return self.lev.distance(sentence1, sentence2)

def is_same_type(idx1, idx2, tokenizer):
    sent1 = tokenizer.decode([idx1])
    sent2 = tokenizer.decode([idx2])
    if sent1.isalpha() and sent2.isalpha():
        return True
    elif sent1.isdigit() and sent2.isdigit():
        return True
    elif sent1.isalnum() and sent2.isalnum():
        return True
    # symbol
    elif sent1.isalnum() == False and sent2.isalnum() == False:
        return True
    return False


def init_mask(mask):
    mask[:3] = 1
    mask[-4:] = 1
    mask[29892] = 1
    return mask