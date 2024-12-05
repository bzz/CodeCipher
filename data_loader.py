import torch 
import torch.utils.data as data
import tokenize
import io
import jsonlines
from transformers import AutoTokenizer

use_cuda = torch.cuda.is_available()

def remove_docstring(source):
    return source[:source.find('"""')] + source[source.rfind('"""') + 3:]

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



class SummaryLlamaData(data.Dataset):
    """
    Dataset that has binary samples.
    """
    def __init__(self, file_name):
        # 1. Initialize file path or list of file names.
        self.data = []
        with jsonlines.open(file_name, 'r') as reader:
            for item in reader:
                self.data.append(item)
        self.tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-hf')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    def __getitem__(self, offset):
        code = self.data[offset]['code']
        text = code.replace(self.data[offset]['docstring'], "<FILL_ME>", 1)
        input = self.tokenizer(text, return_tensors="pt")
        input_ids = input['input_ids'][0]
        input_mask = input['attention_mask'][0]
        docstring = ' '.join(self.data[offset]['docstring_tokens'])
        # if ' .' in docstring:
        #     docstring = docstring[:docstring.find(' .') + 2]
        target = self.tokenizer(docstring, return_tensors="pt")
        target_ids = target['input_ids'][0]
        target_mask = target['attention_mask'][0]
        return input_ids, input_mask, target_ids, target_mask

    def __len__(self):
        return len(self.data)
    
class SummaryLlamaInstrcutData(data.Dataset):
    """
    Dataset that has binary samples.
    """
    def __init__(self, file_name):
        # 1. Initialize file path or list of file names.
        print(file_name)
        self.data = []
        with jsonlines.open(file_name, 'r') as reader:
            for item in reader:
                self.data.append(item)
        self.data = self.data
        self.tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Instruct-hf')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    def __getitem__(self, offset):
        system = 'Generate docstring for the code in 10 words'
        code = ' '.join(self.data[offset]['code_tokens'])
        print(code)
        user = code + '\n"""Please fill this sentence "The goal of this function is to " in 10 words: '

        prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user}[/INST]"
        input = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = input['input_ids'][0]
        input_mask = input['attention_mask'][0]
        docstring = ' '.join(self.data[offset]['docstring_tokens'])
        if '.' in docstring:
            docstring = docstring[:docstring.find('.') + 2]
        elif docstring[-1].isalpha():
            docstring = docstring + ' .'
        target = self.tokenizer(docstring, return_tensors="pt")
        target_ids = target['input_ids'][0]
        target_mask = target['attention_mask'][0]
        return input_ids, input_mask, target_ids, target_mask,  ' '.join(self.data[offset]['code_tokens'])

    def __len__(self):
        return len(self.data)



class SummaryLlamaInstrcutTrainData(data.Dataset):
    """
    Dataset that has binary samples.
    """
    def __init__(self, file_name):
        # 1. Initialize file path or list of file names.
        self.data = []
        with jsonlines.open(file_name, 'r') as reader:
            for item in reader:
                self.data.append(item)
        self.data = self.data
        self.tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Instruct-hf')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    def __getitem__(self, offset):
        def tokenize_dialog(prompt, answer, code_sent, tokenizer):

            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False) 
            answer_tokens = tokenizer.encode(f"{answer} {tokenizer.eos_token}", add_special_tokens=False) 
            dialog_tokens = torch.tensor(prompt_tokens + answer_tokens)
            #Add labels, convert prompt token to -100 in order to ignore in loss function
            labels_tokens = torch.tensor(len(prompt_tokens)*[-100,] + answer_tokens)
            id_ls = ' '.join([str(i) for i in dialog_tokens.numpy()])
            mask_sent = id_ls.replace(code_sent, ' '.join(['-1024',] * len(code_sent.split())))
            mask = torch.tensor([1 if i != '-1024' else 0 for i in mask_sent.split()])
            combined_tokens = {
                "input_ids": dialog_tokens,
                "labels": labels_tokens,
                "mask": mask
            }

            return dict(combined_tokens, attention_mask=torch.tensor([1]*len(combined_tokens["input_ids"])))
        
        system = 'Generate docstring for the code in 10 words. '
        user = ' '.join(self.data[offset]['code_tokens'][:130]) + '\n"""Please fill this sentence "The goal of this function is to " in 10 words: '

        prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user}[/INST]"

        code = '\n' + ' '.join(self.data[offset]['code_tokens'][:130])
        code_sent = self.tokenizer.encode(code, add_special_tokens=False)[2:]
        code_sent = ' '.join([str(i) for i in code_sent])
        
        
        docstring = ' '.join(self.data[offset]['docstring_tokens'])
        if '.' in docstring:
            docstring = docstring[:docstring.find('.') + 2]
        elif docstring[-1].isalpha():
            docstring = docstring + ' .'
        
        item_dict = tokenize_dialog(prompt, docstring, code_sent, self.tokenizer)
        return item_dict

    def __len__(self):
        return len(self.data)

class SummaryLlamaInstrcutTestData(data.Dataset):
    """
    Dataset that has binary samples.
    """
    def __init__(self, file_name):
        with open(file_name, 'r') as reader:
            self.data = reader.readlines()
        self.tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Instruct-hf')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    def __getitem__(self, offset):
        
        code, docstring = self.data[offset][:-1].split('++++++++')

        input = self.tokenizer.convert_tokens_to_ids(code.split(' '))
        input_ids = torch.tensor(input)
        input_mask = input_ids != 0
        
        target = self.tokenizer(docstring, return_tensors="pt")
        target_ids = target['input_ids'][0]
        target_mask = target['attention_mask'][0]
        return input_ids, input_mask, target_ids, target_mask

    def __len__(self):
        return len(self.data)
