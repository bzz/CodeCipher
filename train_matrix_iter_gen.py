import torch
from tqdm import tqdm
from torch import nn
import json
import sys
# torch.cuda.set_device(0)
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

import numpy as np
from utils.utils import init_mask, nn_project, GPT2LM, testSummaryModel, EditDistance
import time
import datetime
import random

from tqdm import tqdm
from data_loader_gen import CodeGenLlamaData, CodeGenLlamaTestData
from modeling_llama import LlamaForCausalLM


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
# get seed_val from input parameter
arguments = sys.argv
if len(arguments) > 1:
    seed_val = int(arguments[1])
    step_size = int(arguments[2])
else:
    seed_val = 114115
    step_size = 32

def save_model(model, step):
    """Save model parameters to checkpoint"""
    os.makedirs(f'obfuscation_codellama_gen/', exist_ok=True)
    ckpt_path=f'obfuscation_codellama_gen/new_model_{step}.pkl'
    print(f'Saving model parameters to {ckpt_path}')
    torch.save(model, ckpt_path)


tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Instruct-hf')
llama_model = LlamaForCausalLM.from_pretrained('codellama/CodeLlama-7b-Instruct-hf')
ppl_model = GPT2LM(cuda = 0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mse_func = torch.nn.MSELoss()
llama_model.train().to(device)


for name, param in llama_model.named_parameters(remove_duplicate = False):

    if  name == 'model.perturb_embeddings':
        param.requires_grad = True
    else:
        param.requires_grad = False



timestamp=datetime.datetime.now().strftime('%Y%m%d%H%M')

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)

epochs = 1


train_set=CodeGenLlamaData('data/gen_data/train_data.jsonl')
valid_set=CodeGenLlamaTestData('data/gen_data/HumanEval.jsonl')
train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=1)
valid_loader=torch.utils.data.DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=1)
print("Loaded data!")
torch.autograd.set_detect_anomaly(True)
total_t0 = time.time()
best_acc = 0
not_increase_num = 0
all_emb = llama_model.model.embed_tokens(torch.arange(llama_model.vocab_size).to(device))
mask = torch.zeros_like(all_emb)
mask = init_mask(mask)



llama_model.model.init_perturb()
total_bleu = 0
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================
    

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    llama_model.train()
    # time for a single epoch
    t0 = time.time()

    # reset total loss for each epoch
    total_train_loss = 0
    ans_ls = []
    ground_truth_ls = []
    for step, batch in enumerate(tqdm(train_loader)):
        # output the progress information
        if step == step_size:
            break

        elapsed = format_time(time.time() - t0)
        print('Loss = {:} Elapsed: {:}.'.format(total_train_loss, elapsed))
        total_train_loss = 0

        


        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        target_ids = batch[2].to(device)
        code_mask = batch[3].to(device)
        if len(input_ids[0]) > 1000:
            continue


        optimizer = torch.optim.AdamW([llama_model.model.perturb_embeddings],
                        lr = 0.002, # args.learning_rate - default is 5e-5
                        eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                        )
        min_loss = 100000
        final_copy_ids = None
        final_copy_emb = None
        flag = 0
        for ii in range(10):
            # reset grad
            llama_model.zero_grad()  
            llama_model.model.perturb_embeddings.data = all_emb * mask + llama_model.model.perturb_embeddings.data * (1 - mask)
            llama_model.model.get_perturb(input_ids)          # .emb = .perturb_embeddings[] lookup
            copy_emb = llama_model.model.emb.clone().detach() # save .emb
                                                              # projects .perturb_embeddings[] on tokens
            project_emb, nn_indices = nn_project(copy_emb, llama_model.model.embed_tokens, input_ids)
            nn_indices = nn_indices * (1 - code_mask[0]) + input_ids[0] * code_mask
            if ii == 0: # prompt & ppl
                org_sent = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                idx1 = org_sent.find('Please complete this code from head: \n') + len('Please complete this code from head: \n')
                idx2 = org_sent.find('please only output the code.')
                org_sent = org_sent[idx1:idx2]
                new_sent = tokenizer.decode(nn_indices[0], skip_special_tokens=True)
                idx1 = new_sent.find('Please complete this code from head: \n') + len('Please complete this code from head: \n')
                idx2 = new_sent.find('please only output the code.')
                new_sent = new_sent[idx1:idx2]
                ppl_org = ppl_model(org_sent)
                ppl_new = ppl_model(new_sent)

            if ii == 0 and ppl_new > ppl_org * 10: # early stop if ppl grows
                flag = 1
                break

            llama_model.model.emb.data = project_emb.data     # .emb = projected 

            loss = llama_model(input_ids = input_ids, labels = target_ids).loss

            org_emb = all_emb[input_ids]
            loss_mse = mse_func(project_emb, org_emb)
            loss = loss - loss_mse # final loss
            if loss < min_loss:
                min_loss = loss
                final_copy_ids = nn_indices
                final_copy_emb = copy_emb 
            # total loss
            total_train_loss += loss.item()
            llama_model.model.emb.data = copy_emb.data       # restore .emb (before projection???)
            # backward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # use mask to make sure the first 5 and last 100 are not changed
            llama_model.model.perturb_embeddings.data = all_emb * mask + llama_model.model.perturb_embeddings.data * (1 - mask)
        
        if flag == 1:
            final_copy_ids = nn_indices
            final_copy_ids = input_ids[0] * code_mask[0] + final_copy_ids * (1 - code_mask[0])
            final_copy_emb = copy_emb
        else:
            final_copy_ids = input_ids[0] * code_mask[0] + final_copy_ids * (1 - code_mask[0])
            for i in range(len(final_copy_ids[0])):
                idx = input_ids[0][i].item()
                if mask[idx][0] == 0:
                    llama_model.model.perturb_embeddings[idx].data = final_copy_emb[0][i]
                if idx != final_copy_ids[0][i].item():
                    mask[idx] = 1
                    all_emb[idx] = final_copy_emb[0][i]
        # clean the gpu
        # torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
        
        # empty cache
        # torch.cuda.empty_cache()        
    
    # time for a single epoach
    training_time = format_time(time.time() - t0)

    print("")
    save_model(llama_model.model.perturb_embeddings, step_size)
    # ========================================
    #               Validation
    # ========================================
    # after each epcoh
    cnt = 0
    print("")
    print("Running Validation...")

    t0 = time.time()

    llama_model.eval()
    # Tracking variables 
    total_eval_accuracy = 0
    total_pred_accuracy = 0
    total_name_ok_acc = 0
    total_eval_loss = 0
    # Evaluate data for one epoch
    cnt = 0
    emb_ans = []
    new_emb_ans = []
    org_ans = []
    ground_truth = []
    org_tokens = []
    new_tokens = []
    org_code_ls = []
    new_code_ls = []
    ppl_ls = []
    result_file_sent = open('result_llama_16_sent_0003.txt', 'w')
    result_file_tok = open('result_llama_16_tok_0003.txt', 'w')
    with open('data/gen_data/HumanEval.jsonl', 'r') as file:
        data = file.readlines()
    for step, batch in enumerate(tqdm(valid_loader)):
        
        cnt += 1
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        target_ids = batch[2].to(device)
        code_mask = batch[3].to(device)
        org_code = batch[4][0]
        org_code_ls.append(org_code)
        # Inference! 
        llama_model.model.get_perturb(input_ids) # set .emb
        copy_emb = llama_model.model.emb.clone().detach()
        project_emb, nn_indices = nn_project(copy_emb, llama_model.model.embed_tokens, input_ids)
        nn_indices = nn_indices * (1 - code_mask[0]) + input_ids[0] * code_mask

        org_tokens.extend(input_ids.cpu().numpy())
        new_tokens.append(list(nn_indices.cpu().numpy()[0]))
        new_code = tokenizer.decode(nn_indices[0], skip_special_tokens=True)
        data[step] = json.loads(data[step])
        data[step]['new_input'] = new_code
        idx1 = new_code.find('Please complete this code from head: \n') + len('Please complete this code from head: \n')
        idx2 = new_code.find('please only output the code.')
        only_code = new_code[idx1:idx2]
        print(only_code)

        ppl_ls.append(ppl_model(only_code))
        new_code_ls.append(only_code)

    del llama_model
    with open('HumanEval1.jsonl', 'w') as file:
        for i in range(len(data)):
            file.write(json.dumps(data[i]) + '\n')
    test_model = testSummaryModel('codellama/CodeLlama-7b-Instruct-hf', task="Completion")
    test_model(new_tokens, len(new_tokens))    
    # print the mean ppl


    print(np.mean(ppl_ls))
    editDist = EditDistance(normalized=True)
    edit_ls = []
    for i in range(len(org_code_ls)):
        edit_ls.append(editDist(org_code_ls[i], new_code_ls[i]))

    print(np.mean(edit_ls))


print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
