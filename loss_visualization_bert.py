"""
The majority of this code is taken from:

https://github.com/tomgoldstein/loss-landscape

@inproceedings{visualloss,
  title={Visualizing the Loss Landscape of Neural Nets},
  author={Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom},
  booktitle={Neural Information Processing Systems},
  year={2018}
}

We have merely adapted it to plot the loss landscape for BERT
"""


import h5py
import pickle
import torch
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--seed")
    parser.add_argument("-p", "--training_set_pct")
    parser.add_argument("-t", "--perturbation")
    args = parser.parse_args()
    parser_params = vars(args)

    torch.manual_seed(int(parser_params['seed']))

    with open('./Fisher/BERT/fisher_bert.pkl','rb') as fp:
        fisher = pickle.load(fp)
    
    fisher_means = []
    for i in range(len(fisher)):
        fisher_means.append(np.mean(fisher[i].numpy()))
    
    fisher_inds = np.argsort(fisher_means)
    fisher_inds_top = fisher_inds[101:202]
    fisher_inds_bottom = fisher_inds[:101]

    wikiset = load_dataset('wikipedia', '20220301.simple')
    indices = int(wikiset.num_rows['train'] * (int(parser_params['training_set_pct'])/100))
    wikiset = wikiset['train'].shuffle(seed=int(parser_params['seed'])).select(range(indices))

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def encode_with_truncation(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    train_dataset = wikiset.map(encode_with_truncation, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)    

    training_args = TrainingArguments(
        output_dir = './',
        do_train = False,
        do_eval = True,
        prediction_loss_only = True,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 32)

    print('Dataset tokenized.')

    bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
    w = [x.data for x in bert.parameters()]

    xmin = -1.5
    xmax = 1.5
    xnum = 16
    ymin = -1.5
    ymax = 1.5
    ynum = 16

    def normalize_direction(direction, weights):
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))

    def create_random_dimension(weights):
        direction = [torch.randn(w.size()) for w in weights]
        ignore = 'biasbn'
        for d, w in zip(direction, weights):
            if d.dim() <= 1:
                if ignore == 'biasbn':
                    d.fill_(0)
                else:
                    d.copy_(w)
            else:
                normalize_direction(d, w)
        return direction


    def write_list(f, name, direction):
        grp = f.create_group(name)
        for i, l in enumerate(direction):
            if isinstance(l, torch.Tensor):
                l = l.numpy()
            grp.create_dataset(str(i), data=l)

    def read_list(f, name):
        grp = f[name]
        return [grp[str(i)] for i in range(len(grp))]

    def setup_dir_file(dir_file):
        f = h5py.File(dir_file,'w')
        x_direction = create_random_dimension(w)
        write_list(f, 'xdirection', x_direction)
        y_direction = create_random_dimension(w)
        write_list(f, 'ydirection', y_direction)
        f.close()


    setup_dir_file('dir_res256.h5')

    def setup_surface_file(surf_file, dir_file):
        f = h5py.File(surf_file, 'a')
        f['dir_file'] = dir_file
        xcoordinates = np.linspace(xmin, xmax, num=xnum)
        f['xcoordinates'] = xcoordinates
        ycoordinates = np.linspace(ymin, ymax, num=ynum)
        f['ycoordinates'] = ycoordinates
        f.close()
        return surf_file

    surf_file = setup_surface_file('surf_res256.h5','dir_res256.h5')

    f = h5py.File('surf_res256.h5', 'r+')

    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None
    shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
    losses = -np.ones(shape=shape)

    def get_unplotted_indices(vals, xcoordinates, ycoordinates=None):
        inds = np.array(range(vals.size))
        inds = inds[vals.ravel() <= 0]
        if ycoordinates is not None:
            xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
            s1 = xcoord_mesh.ravel()[inds]
            s2 = ycoord_mesh.ravel()[inds]
            return inds, np.c_[s1,s2]
        else:
            return inds, xcoordinates.ravel()[inds]

    def split_inds(num_inds, nproc):
        chunk = num_inds // nproc
        remainder = num_inds % nproc
        splitted_idx = []
        for rank in range(0, nproc):
            start_idx = rank * chunk + min(rank, remainder)
            stop_idx = start_idx + chunk + (rank < remainder)
            splitted_idx.append(range(start_idx, stop_idx))

        return splitted_idx

    def get_job_indices(vals, xcoordinates, ycoordinates, comm):

        inds, coords = get_unplotted_indices(vals, xcoordinates, ycoordinates)
        rank = 0
        nproc = 1
        splitted_idx = split_inds(len(inds), nproc)
        inds = inds[splitted_idx[rank]]
        coords = coords[splitted_idx[rank]]
        inds_nums = [len(idx) for idx in splitted_idx]

        return inds, coords, inds_nums

    inds, coords, inds_nums = get_job_indices(losses, xcoordinates, ycoordinates, None)
    losses = []

    def load_directions(dir_file):
        f = h5py.File(dir_file, 'r')
        xdirection = read_list(f, 'xdirection')
        ydirection = read_list(f, 'ydirection')
        directions = [xdirection, ydirection]
        return directions

    d = load_directions('dir_res256.h5')

    print('Directions and Surface files setup.')

    def set_weights(net, weights, directions=None, step=None):
        dx = directions[0]
        dy = directions[1]
        changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        indx = 0
        for (p, w, d) in zip(net.parameters(), weights, changes):
            if str(parser_params['perturbation']) == 'top':
                if indx in fisher_inds_top:
                    p.data = w + torch.Tensor(d).type(type(w))
                indx += 1
            elif str(parser_params['perturbation']) == 'bottom':
                if indx in fisher_inds_bottom:
                    p.data = w + torch.Tensor(d).type(type(w))
                indx += 1
            else:
                p.data = w + torch.Tensor(d).type(type(w))

    for count, ind in tqdm(enumerate(inds)):
        coord = coords[count]
        bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        set_weights(bert, w, d, coord)
        trainer = Trainer(
            model=bert,
            args=training_args,
            data_collator=data_collator,
            eval_dataset=train_dataset
        )
        metrics = trainer.evaluate()
        losses.append(metrics['eval_loss'])

    f['train_loss'] = losses
    f.close()

if __name__ == "__main__":
    main()