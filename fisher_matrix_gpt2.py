import os

if not os.path.exists('./Fisher/GPT2/Gradients'):
    os.makedirs('./Fisher/GPT2/Gradients')

import torch
import pickle
import argparse
from glob import glob
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, DataCollatorForLanguageModeling, set_seed

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--seed")
    parser.add_argument("-p", "--training_set_pct")
    args = parser.parse_args()
    parser_params = vars(args)

    set_seed(int(parser_params['seed']))

    # Load the model, dataset, and tokenize the dataset

    webcrawl = load_dataset('Bingsu/openwebtext_20p')

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model = GPT2LMHeadModel.from_pretrained('gpt2')

    def encode_with_truncation(examples):
        return tokenizer(examples['text'], truncation=True)

    indices = int(webcrawl.num_rows['train'] * (int(parser_params['training_set_pct'])/100)) 
    train_dataset = webcrawl['train'].shuffle(seed=int(parser_params['seed'])).select(range(indices))
    train_dataset = train_dataset.map(encode_with_truncation, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)    

    # Choose compute device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    """
    Here, the loop below computes the Fisher score as formulated in equation (3) in the paper.
    The if len(sqr_gradients) == 100 clause is to store squared gradients for 100 samples at a timr and clear the cache as not to overload the system memory.
    """
    ssg = []
    sqr_gradients = []
    sum_sqr_gradients = []
    individual_gradients = []

    file_idx = 0

    for i in tqdm(train_dataset):
        batch = data_collator([i])
        batch.to(device)
        out = model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'], labels=batch['labels'])
        grads = torch.autograd.grad(out['loss'], model.parameters())
        sqr_gradient = [x.cpu() ** 2 for x in grads]
        individual_gradients.append(torch.mean(torch.cat([tensor.flatten() for tensor in sqr_gradient])).item())
        sqr_gradients.append(sqr_gradient)
        if len(sqr_gradients) == 100:
            for i in range(len(sqr_gradients[0])):
                items = [item[i] for item in sqr_gradients]
                sum_sqr_gradients.append(torch.mul(torch.sum(torch.stack(items),0), 1/indices))
            ssg.append(sum_sqr_gradients)
            del sqr_gradients
            del items
            del sum_sqr_gradients
            sqr_gradients = []
            sum_sqr_gradients = []
        if len(ssg) == 100:
            with open('./Fisher/GPT2/Gradients/ssg_{}.pkl'.format(file_idx), 'wb') as fp:
                pickle.dump(ssg, fp)
            del ssg
            ssg = []
            file_idx += 1
        del batch
        del out
        del grads
        torch.cuda.empty_cache()

    with open('./Fisher/GPT2/Gradients/individual_gradients.pkl','wb') as fp:
        pickle.dump(individual_gradients, fp)

    del individual_gradients

    with open('./Fisher/GPT2/Gradients/ssg_{}.pkl'.format(file_idx),'wb') as fp:
        pickle.dump(ssg, fp)
        
    files = glob('./Fisher/GPT2/Gradients/ssg_*')
    fisher_outer = []
    ssgs = []
    for file in tqdm(files):
        with open(file, 'rb') as fp:
            ssg = pickle.load(fp)
        for i in range(len(ssg[0])):
            items = [item[i] for item in ssg]
            ssgs.append(torch.sum(torch.stack(items),0))
        fisher_outer.append(ssgs)
        del ssg
        del ssgs
        ssgs = []
        
    fisher_inner = []
    for i in tqdm(range(len(fisher_outer[0]))):
        items = [item[i] for item in fisher_outer]
        fisher_inner.append(torch.sum(torch.stack(items),0))
        
    with open('./Fisher/GPT2/fisher_gpt2.pkl','wb') as fp:
        pickle.dump(fisher_inner, fp)

if __name__ == "__main__":
    main()