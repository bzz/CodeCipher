
from dataclasses import replace

import torch 
import torch.utils.data as data
import tokenize
import io
import jsonlines
from transformers import AutoTokenizer
from utils.utils import my_find_java_ide, switch_ide, replace_ide

use_cuda = torch.cuda.is_available()


def remove_one_line_annotation(source):
    source_ls = source.split('\n')
    new_source = []
    for line in source_ls:
        if '#' not in line:
            new_source.append(line)
    return '\n'.join(new_source)

def remove_docs(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out


class CodeTranslateLlamaData(data.Dataset):
    """
    Dataset that has binary samples.
    """
    def __init__(self, file_name):
        # 1. Initialize file path or list of file names.
        self.data = []
        with jsonlines.open(file_name, 'r') as reader:
            for item in reader:
                self.data.append(item)
        self.tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Instruct-hf')
    def refine_code(self, code):
        code_ls = code.split(' ')
        # count for INDENT
        cnt = 0
        new_code_ls = []
        for i in range(len(code_ls)):
            if code_ls[i] == 'NEW_LINE':
                new_code_ls.append('\n')
                if i + 1 < len(code_ls):
                    if code_ls[i + 1] == 'INDENT':
                        cnt += 1
                    elif code_ls[i + 1] == 'DEDENT':
                        cnt -= 1
                new_code_ls.append('\t' * cnt)
            elif code_ls[i] == 'INDENT' or code_ls[i] == 'DEDENT':
                continue
            else:
                new_code_ls.append(' ' + code_ls[i])
        return ''.join(new_code_ls)
    def __getitem__(self, offset):
        # input = tokenizer(remove_docs(self.data[offset]['code']), return_tensors="pt", padding="max_length", truncation=True, max_length=256)
        # print(self.data[offset])
        code = self.data[offset]['java']
        target_code = self.data[offset]['python']
        prompt = f'<s> [INST] Please translate this java code to python: \n {code} \n Please use a functional programming style \n [/INST]'
        target_tokens = target_code + self.tokenizer.eos_token

        input = self.tokenizer.encode(prompt, add_special_tokens=False)
        target = self.tokenizer.encode(target_tokens, add_special_tokens=False)
        input_ids =  torch.tensor(input + target)
        input_mask = torch.tensor([1] * len(input_ids))
        labels_ids = torch.tensor(len(input)*[-100,] + target)

        # sent_org_ls = self.tokenizer.encode(prompt + ' ' + target_tokens, add_special_tokens=False)
        sent_org_ls = []
        for i in range(len(input_ids)):
            sent_org_ls.append(input_ids[i].item())
        code_ls = self.tokenizer.encode(code, add_special_tokens=False)[1:]
        code_sent = ' '.join([str(token) for token in code_ls])
        sent_org = ' '.join([str(token) for token in sent_org_ls])
        if code_sent not in sent_org:
            print('code')
            print(code_sent)
            print('sent_org')
            print(sent_org)
        sent_org = sent_org.replace(code_sent, ' '.join(['-1' for i in range(len(code_ls))]), 1)
        mask = torch.tensor([0 if token == '-1' else 1 for token in sent_org.split()])
        return input_ids, input_mask, labels_ids, mask

    def __len__(self):
        return len(self.data)


class CodeTranslateLlamaIdentifierData(data.Dataset):
    """
    Dataset that has binary samples.
    """
    def __init__(self, file_name):
        # 1. Initialize file path or list of file names.
        self.data = []
        with jsonlines.open(file_name, 'r') as reader:
            for item in reader:
                self.data.append(item)
        self.tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Instruct-hf')
    def refine_code(self, code):
        code_ls = code.split(' ')
        # count for INDENT
        cnt = 0
        new_code_ls = []
        for i in range(len(code_ls)):
            if code_ls[i] == 'NEW_LINE':
                new_code_ls.append('\n')
                if i + 1 < len(code_ls):
                    if code_ls[i + 1] == 'INDENT':
                        cnt += 1
                    elif code_ls[i + 1] == 'DEDENT':
                        cnt -= 1
                new_code_ls.append('\t' * cnt)
            elif code_ls[i] == 'INDENT' or code_ls[i] == 'DEDENT':
                continue
            else:
                new_code_ls.append(' ' + code_ls[i])
        return ''.join(new_code_ls)
    def __getitem__(self, offset):
        code = self.data[offset]['java']
        target_code = self.data[offset]['python']
        prompt = f'<s> [INST] Please translate this java code to python: \n {code} \n Please use a functional programming style \n [/INST]'
        target_tokens = target_code + self.tokenizer.eos_token

        input = self.tokenizer.encode(prompt, add_special_tokens=False)
        target = self.tokenizer.encode(target_tokens, add_special_tokens=False)
        input_ids =  torch.tensor(input + target)
        input_mask = torch.tensor([1] * len(input_ids))
        labels_ids = torch.tensor(len(input)*[-100,] + target)

        sent_org_ls = []
        for i in range(len(input_ids)):
            sent_org_ls.append(input_ids[i].item())
        code_ls = self.tokenizer.encode(code, add_special_tokens=False)[1:]


        varnames, last_symbols = my_find_java_ide(code, last_symbol=True)

        idict_mask = switch_ide(varnames, flag='mask', last_symbols=last_symbols)
        masked_code = replace_ide(code, idict_mask, language='java', split= False)
        new_code = self.tokenizer.tokenize(prompt + target_tokens)
        new_prompt = f'<s> [INST] Please translate this java code to python: \n {masked_code} \n Please use a functional programming style \n [/INST]'
        masked_code = self.tokenizer.tokenize(new_prompt + target_tokens)
        new_masked_code = []
        for i in range(len(masked_code)):
            if i != 0 and masked_code[i] == '▁B' and masked_code[i - 1] == '<s>':
                continue
            else:
                new_masked_code.append(masked_code[i])
        print(len(new_masked_code))
        print(len(new_code))
        mask = torch.tensor([0 if new_code[i] != new_masked_code[i] else 1 for i in range(len(new_masked_code))])

        return input_ids, input_mask, labels_ids, mask

    def __len__(self):
        return len(self.data)
    


class CodeTranslateLlamaTestData(data.Dataset):
    """
    Dataset that has binary samples.
    """
    def __init__(self, model_input_file_name, model_output_file_name):
        # 1. Initialize file path or list of file names.
        self.org_data = []
        with jsonlines.open(model_input_file_name, 'r') as reader:
            for item in reader:
                self.org_data.append(item)
        with jsonlines.open(model_output_file_name, 'r') as reader:
            self.tgt_data = []
            for item in reader:
                self.tgt_data.append(item)

        self.tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-13b-Instruct-hf')
    def __getitem__(self, offset):

        code = self.org_data[offset]['declaration'] + self.org_data[offset]['canonical_solution']
        origin_code = code
        prompt = f'<s> [INST] Please translate this java code to python: \n {code} \n Please use a functional programming style \n [/INST]'
        target_tokens = self.tgt_data[offset]['declaration'] + self.tgt_data[offset]['canonical_solution'] + self.tokenizer.eos_token

        input = self.tokenizer.encode(prompt, add_special_tokens=False)
        target = self.tokenizer.encode(target_tokens, add_special_tokens=False)
        input_ids =  torch.tensor(input)
        input_mask = torch.tensor([1] * len(input_ids))
        labels_ids = torch.tensor(target)
        

        sent_org_ls = self.tokenizer.encode(prompt, add_special_tokens=False)
        code_ls = self.tokenizer.encode(code, add_special_tokens=False)[1:]

        code_sent = ' '.join([str(token) for token in code_ls])
        sent_org = ' '.join([str(token) for token in sent_org_ls])
        if code_sent not in sent_org:
            print('code')
            print(code_sent)
            print('sent_org')
            print(sent_org)
        sent_org = sent_org.replace(code_sent, ' '.join(['-1' for i in range(len(code_ls))]), 1)
        mask = torch.tensor([0 if token == '-1' else 1 for token in sent_org.split()])
        return input_ids, input_mask, labels_ids, mask, origin_code

    def __len__(self):
        return len(self.org_data)
