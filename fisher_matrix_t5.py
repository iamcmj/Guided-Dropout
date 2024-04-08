import os

if not os.path.exists('./Fisher/T5/Gradients'):
    os.makedirs('./Fisher/T5/Gradients')

import torch
import random
import pickle
import argparse
from glob import glob
from tqdm import tqdm
from datasets import load_dataset
from transformers import T5TokenizerFast, T5ForConditionalGeneration, set_seed

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--seed")
    parser.add_argument("-p", "--training_set_pct")
    args = parser.parse_args()
    parser_params = vars(args)

    set_seed(int(parser_params['seed']))

    # Load the model, dataset, and tokenize the dataset

    c4 = load_dataset('crumb/C4-K8-1M')
    indices = int(c4.num_rows['train'] * (int(parser_params['training_set_pct'])/100)) 
    train_dataset = c4['train'].shuffle(seed=int(parser_params['seed'])).select(range(indices))

    tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    sentinel_tokens = tokenizer.additional_special_tokens

    def encode_with_truncation(examples):
        input_tokenized = tokenizer.tokenize(examples['text'], truncation = True)
        num_tokens_to_mask = round(0.15 * len(input_tokenized))
        masking_indices = []
        j = None
        seen_list = [None]
        for i in range(num_tokens_to_mask):
            while j in seen_list:
                j = random.randint(0, len(input_tokenized))
            seen_list.append(j)
            masking_indices.append(j)

        input_seq = []
        target_seq = []
        sentinel_id = 0
        for idx, val in enumerate(input_tokenized):
            if idx in masking_indices:
                input_seq.append(sentinel_tokens[sentinel_id])
                target_seq.append(sentinel_tokens[sentinel_id])
                target_seq.append(val)
                sentinel_id += 1
            else:
                input_seq.append(val)
        
        input_encodings = tokenizer(input_seq, is_split_into_words = True, truncation = True)
        target_encodings = tokenizer(target_seq, is_split_into_words = True, truncation = True)
        
        encodings = {
            'input_ids': input_encodings['input_ids'], 
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids'],
            'decoder_attention_mask': target_encodings['attention_mask']
        }

        return encodings

    train_dataset = train_dataset.map(encode_with_truncation)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask'])

    model = T5ForConditionalGeneration.from_pretrained('t5-base')

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
        out = model(input_ids = i['input_ids'].reshape(1, i['input_ids'].shape[0]).to(device), \
                    attention_mask = i['attention_mask'].reshape(1, i['attention_mask'].shape[0]).to(device),\
                    labels = i['labels'].reshape(1, i['labels'].shape[0]).to(device),\
                    decoder_attention_mask = i['decoder_attention_mask'].reshape(1, \
                    i['decoder_attention_mask'].shape[0]).to(device))
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
            with open('./Fisher/T5/Gradients/ssg_{}.pkl'.format(file_idx), 'wb') as fp:
                pickle.dump(ssg, fp)
            del ssg
            ssg = []
            file_idx += 1
        del batch
        del out
        del grads
        torch.cuda.empty_cache()

    with open('./Fisher/T5/Gradients/individual_gradients.pkl','wb') as fp:
        pickle.dump(individual_gradients, fp)

    del individual_gradients

    with open('./Fisher/T5/Gradients/ssg_{}.pkl'.format(file_idx),'wb') as fp:
        pickle.dump(ssg, fp)
        
    files = glob('./Fisher/T5/Gradients/ssg_*')
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
        
    with open('./Fisher/T5/fisher_t5.pkl','wb') as fp:
        pickle.dump(fisher_inner, fp)

if __name__ == "__main__":
    main()