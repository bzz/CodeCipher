
from dataclasses import replace

import torch 
import torch.utils.data as data
import tokenize
import io
import jsonlines
from transformers import AutoTokenizer

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


class CodeGenData(data.Dataset):
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
    def __getitem__(self, offset):
        line_num = self.data[offset]['canonical_solution'].split('\n')
        prompt_code = '\n'.join(line_num[:len(line_num) // 2])
        code = self.data[offset]['prompt'] + prompt_code
        prompt = f'<s> [INST] Please complete this code from head: \n {code} \nplease only output the code. Please complete this code from head:[/INST]'
        target_tokens = self.data[offset]['prompt'] + self.data[offset]['canonical_solution'] + self.tokenizer.eos_token
        input = self.tokenizer.encode(prompt, add_special_tokens=False)
        target = self.tokenizer.encode(target_tokens, add_special_tokens=False)
        input_ids =  torch.tensor(input + target)
        input_mask = torch.tensor([1] * len(input_ids))
        labels_ids = torch.tensor(len(input)*[-100,] + target)
        # make a mask
        code = self.tokenizer.decode(input_ids)
        # def xxx :
        left1 = code.find('def')
        right1 = code.find(":")
        func_signature = code[left1:right1 + 1]
        half_code = prompt_code
        
        func_signature_ls = self.tokenizer.encode(' ' + func_signature, add_special_tokens=False)
        half_code_ls = self.tokenizer.encode(half_code, add_special_tokens=False)

        func_signature_sent = ' '.join([str(token) for token in func_signature_ls])
        half_code_sent = ' '.join([str(token) for token in half_code_ls])
        sent_org = ' '.join([str(token) for token in input_ids])
        sent_org = sent_org.replace(func_signature_sent, ' '.join(['-1' for i in range(len(func_signature_sent))]), 1)
        sent_org = sent_org.replace(half_code_sent, ' '.join(['-1' for i in range(len(half_code_sent))]), 1)

        mask = torch.tensor([0 if token == '-1' else 1 for token in sent_org.split()])

        return input_ids, input_mask, labels_ids, mask

    def __len__(self):
        return len(self.data)

    
class CodeGenLlamaData(data.Dataset):
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
    def get_half_code(self, code):
        half = code.rfind('"""\n')
        right_half = code[half + 4:]
        line_num = right_half.split('\n')
        half_num = (len(line_num) -1) // 2
        total_line = code.split('\n')
        
        prompt_code = '\n'.join(total_line[:(len(total_line) - len(line_num) + half_num)])
        ans_code = '\n'.join(line_num[:half_num])
        return prompt_code, ans_code
    def __getitem__(self, offset):
        # org_code = remove_one_line_annotation(self.data[offset]['code'])
        # code, half_code = self.get_half_code(org_code)
        
        line_num = self.data[offset]['canonical_solution'].split('\n')
        prompt_code = '\n'.join(line_num[:(len(line_num) - 1) // 2])
        code = self.data[offset]['prompt'] + prompt_code
        half_code = prompt_code
        org_code = code
        prompt = f'<s> [INST] Please complete this code from head: \n {code} \n\n please only output the code. Please complete this code from head: [/INST]  Sure, here is the completed code:\n```\n'
        target_tokens = org_code + self.tokenizer.eos_token
        input = self.tokenizer.encode(prompt, add_special_tokens=False)
        target = self.tokenizer.encode(target_tokens, add_special_tokens=False)
        input_ids =  torch.tensor(input + target)
        input_mask = torch.tensor([1] * len(input_ids))
        labels_ids = torch.tensor(len(input)*[-100,] + target)
        # make a mask
        code = self.tokenizer.decode(input_ids)
        sent_org_ls = self.tokenizer.encode(code, add_special_tokens=False)
        # def xxx :
        left1 = code.find('def')
        right1 = code.find(":\n", left1)
        func_signature = code[left1:right1 + 1]
        func_signature_ls = self.tokenizer.encode(func_signature, add_special_tokens=False)[1:]
        half_code_ls = self.tokenizer.encode(' ' + half_code, add_special_tokens=False)[1:]

        func_signature_sent = ' '.join([str(token) for token in func_signature_ls])
        half_code_sent = ' '.join([str(token) for token in half_code_ls])
        sent_org = ' '.join([str(token) for token in sent_org_ls])
        if func_signature_sent not in sent_org:
            print('func_signature_sent')
            print(func_signature_sent)
            print('sent_org')
            print(sent_org)
        # sent_org = sent_org.replace(func_signature_sent, ' '.join(['-1' for i in range(len(func_signature_ls))]), 1)
        if half_code_sent not in sent_org:
            print('half_code_sent')
            print(half_code_sent)
            print('sent_org')
            print(sent_org)
        sent_org = sent_org.replace(half_code_sent, ' '.join(['-1' for i in range(len(half_code_ls))]), 1)
        mask = torch.tensor([0 if token == '-1' else 1 for token in sent_org.split()])
        return input_ids, input_mask, labels_ids, mask

    def __len__(self):
        return len(self.data)
    

class CodeGenLlamaTestData(data.Dataset):
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
    def __getitem__(self, offset):

        line_num = self.data[offset]['canonical_solution'].split('\n')
        prompt_code = '\n'.join(line_num[:(len(line_num) - 1) // 2])
        prompt_code = self.data[offset]['canonical_solution']
        code = self.data[offset]['prompt'] + prompt_code

        final_output_code = code

        prompt = f'''<s> [INST] Please complete this code from head: \n {code} \n\n please only output the code. Please complete this code from head: [/INST]'''


        target_tokens = self.data[offset]['prompt'] + self.data[offset]['canonical_solution'] + self.tokenizer.eos_token

        input = self.tokenizer.encode(prompt, add_special_tokens=False)
        target = self.tokenizer.encode(target_tokens, add_special_tokens=False)
        input_ids =  torch.tensor(input)
        input_mask = torch.tensor([1] * len(input_ids))
        labels_ids = torch.tensor(target)
        # make a mask
        code = self.tokenizer.decode(input_ids)
        sent_org_ls = self.tokenizer.encode(code, add_special_tokens=False)
        # def xxx :
        left1 = code.find('def')
        right1 = code.find(":\n", left1)
        func_signature = code[left1:right1 + 1]
        half_code = prompt_code
        func_signature_ls = self.tokenizer.encode('\n' + func_signature, add_special_tokens=False)[2:]
        half_code_ls = self.tokenizer.encode(' ' + half_code, add_special_tokens=False)[1:]

        func_signature_sent = ' '.join([str(token) for token in func_signature_ls])
        half_code_sent = ' '.join([str(token) for token in half_code_ls])
        sent_org = ' '.join([str(token) for token in sent_org_ls])
        if func_signature_sent not in sent_org:
            print('func_signature_sent')
            print(func_signature_sent)
            print('sent_org')
            print(sent_org)
        sent_org = sent_org.replace(func_signature_sent, ' '.join(['-1' for i in range(len(func_signature_ls))]), 1)
        if half_code_sent not in sent_org:
            print('half_code_sent')
            print(half_code_sent)
            print('sent_org')
            print(sent_org)
        sent_org = sent_org.replace(half_code_sent, ' '.join(['-1' for i in range(len(half_code_ls))]), 1)

        mask = torch.tensor([0 if token == '-1' else 1 for token in sent_org.split()])
        return input_ids, input_mask, labels_ids, mask, final_output_code

    def __len__(self):
        return len(self.data)
    
