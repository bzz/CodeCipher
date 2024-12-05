import os
import re
import bleu
import torch
import evaluate
import numpy as np
from pathlib import Path
from tree_sitter import Language, Parser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

def calculate_bleu(predictions, references, output_dir):
    dir_ = Path(__file__).parent / output_dir
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    p = []
    with open(os.path.join(dir_, "test_0.gold"), 'w', encoding='utf-8') as f:
        for idx, (predict, target) in enumerate(zip(predictions, references)):
            p.append(str(idx) + '\t' + predict)
            f.write(str(idx) + '\t' + target + '\n')

    (goldMap, predictionMap) = bleu.computeMaps(p, os.path.join(dir_, "test_0.gold"))
    test_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    return test_bleu


def calculate_bleu_hf(predictions, references):
    bleu_hf = evaluate.load('bleu')
    bleu_result = bleu_hf.compute(predictions=predictions, references=references, smooth=True)['bleu']
    return bleu_result


def calculate_rouge_hf(predictions, references):
    for i in range(len(predictions)):
        predictions[i] = predictions[i][0]
    for i in range(len(references)):
        references[i] = references[i][0]
    rouge_hf = evaluate.load('rouge')
    rouge_result = rouge_hf.compute(predictions=predictions, references=references, use_aggregator=True)[
        'rougeL']
    return rouge_result


def calculate_meteor_hf(predictions, references):
    for i in range(len(predictions)):
        predictions[i] = predictions[i][0]
    for i in range(len(references)):
        references[i] = references[i][0]
    meteor_hf = evaluate.load('meteor')
    meteor_result = meteor_hf.compute(predictions=predictions, references=references)['meteor']
    return meteor_result


