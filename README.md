# Guided-Dropout
This is the anonymously hosted source code repository for the COLM 2024 submission 'Information Guided Regularization for Fine-tuning Language Models'.

Before using this repository, please make sure you have the following packages installed in your environment:
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Tqdm](https://github.com/tqdm/tqdm)
- [Sklearn](https://scikit-learn.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Jupyter Notebook](https://jupyter.org/)
- [PyTorch](https://pytorch.org/)
- [Huggingface Transformers & Dataset](https://huggingface.co/)

This environment can be built simply using the requirements file supplied in this repository. Inside your virtual environment, please run:

```
pip install -r requirements.txt
```

> **Note:** The versions of PyTorch (and subsequently Huggingface) required for GPU-use is highly dependent on the user GPU and CUDA version.

Please follow the step-wise description below to replicate the results published in 'Information Guided Regularization for Fine-tuning Language Models'.

## Computing and Visualizing Fisher Information Matrices
The *fisher_matrix_{model}.py* scripts compute the empirical Fisher (eqn 3 in the paper) and take in the randomization seed and training set percentage as arguments:

```
python fisher_matrix_bert.py -seed 42 --training_set_pct 100
python fisher_matrix_bert.py -seed 1 --training_set_pct 50
```

```
python fisher_matrix_gpt2.py -seed 42 --training_set_pct 100
python fisher_matrix_gpt2.py -seed 1 --training_set_pct 50
```

```
python fisher_matrix_t5.py -seed 42 --training_set_pct 100
python fisher_matrix_t5.py -seed 1 --training_set_pct 50
```
The python scripts will build the required subdirectories under the current directory for you.

The notebooks *Fisher Visualizations.ipynb* and *Fisher Visualizations (Normalized).ipynb* will help you visualize various model based on the Fisher scores that you have computed.

## Computing and Visualizing the Loss Landscape of BERT
The *loss_visualization_bert.py* script computes the loss landscape of the base BERT model. The script takes in 3 arguments for the randomization seed, training set percentage, and perturbation type.

As showcased in Figure 1 of the paper, the perturbation is done either to:
- the entire model (--perturbation normal)
- the top 50% of the parameters with the highest Fisher scores (--perturbation top)
- the bottom 50% of the parameters with the lowest Fisher scores (--perturbation bottom)

If you so choose to, you can compute the surf files representing the 3D loss landscape of base BERT based on any configuration of the 3 arguments. As examples:

```
python loss_visualization_bert.py -seed 42 --training_set_pct 100 --perturbation top
python loss_visualization_bert.py -seed 42 --training_set_pct 100 --perturbation bottom
python loss_visualization_bert.py -seed 42 --training_set_pct 100 --perturbation normal
```

This is a costly process. Thus, precomputed surf files for the above 3 commands can be found in the 'Surfs' directory.

The notebook *Loss Landscape Visualization.ipynb* will help you visualize the 3D loss landscapes constructed from those surf files.

## Evaluating Guided-Dropout and Baselines on GLUE tasks

> **Note:** A [source installation](https://huggingface.co/transformers/v2.9.1/examples.html) of Huggingface transformers is required for running GLUE evaluations.

The *glue_bert_{model}.py* scripts evaluate both our model and the baselines of the glue tasks. The scripts take 4 arguments, the GLUE task, the output directory with the evaluation metrics and models are saved, the training seed, and finally the training set percentage. If the output directory does not exist, the script will create the directory for you. Finally, the GLUE task argument can take values in {cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb, wnli}. 

Thus, running the evaluation script is as simple as:

```
python glue_bert_ours.py --task cola --output_dir ./CoLA -seed 42 --training_set_pct 100
python glue_bert_ours.py --task mnli --output_dir ./MNLI -seed 42 --training_set_pct 100
python glue_bert_ours.py --task mrpc --output_dir ./MRPC -seed 42 --training_set_pct 100
python glue_bert_ours.py --task qnli --output_dir ./QNLI -seed 42 --training_set_pct 100
python glue_bert_ours.py --task qqp --output_dir ./QQP -seed 42 --training_set_pct 100
python glue_bert_ours.py --task rte --output_dir ./RTE -seed 42 --training_set_pct 100
python glue_bert_ours.py --task sst2 --output_dir ./SST2 -seed 42 --training_set_pct 100
python glue_bert_ours.py --task stsb --output_dir ./STSB -seed 42 --training_set_pct 100
python glue_bert_ours.py --task wnli --output_dir ./WNLI -seed 42 --training_set_pct 100
```

```
python glue_bert_dropout.py --task cola --output_dir ./CoLA -seed 42 --training_set_pct 100
python glue_bert_dropout.py --task mnli --output_dir ./MNLI -seed 42 --training_set_pct 100
python glue_bert_dropout.py --task mrpc --output_dir ./MRPC -seed 42 --training_set_pct 100
python glue_bert_dropout.py --task qnli --output_dir ./QNLI -seed 42 --training_set_pct 100
python glue_bert_dropout.py --task qqp --output_dir ./QQP -seed 42 --training_set_pct 100
python glue_bert_dropout.py --task rte --output_dir ./RTE -seed 42 --training_set_pct 100
python glue_bert_dropout.py --task sst2 --output_dir ./SST2 -seed 42 --training_set_pct 100
python glue_bert_dropout.py --task stsb --output_dir ./STSB -seed 42 --training_set_pct 100
python glue_bert_dropout.py --task wnli --output_dir ./WNLI -seed 42 --training_set_pct 100
```

```
python glue_bert_gaussian.py --task cola --output_dir ./CoLA -seed 42 --training_set_pct 100
python glue_bert_gaussian.py --task mnli --output_dir ./MNLI -seed 42 --training_set_pct 100
python glue_bert_gaussian.py --task mrpc --output_dir ./MRPC -seed 42 --training_set_pct 100
python glue_bert_gaussian.py --task qnli --output_dir ./QNLI -seed 42 --training_set_pct 100
python glue_bert_gaussian.py --task qqp --output_dir ./QQP -seed 42 --training_set_pct 100
python glue_bert_gaussian.py --task rte --output_dir ./RTE -seed 42 --training_set_pct 100
python glue_bert_gaussian.py --task sst2 --output_dir ./SST2 -seed 42 --training_set_pct 100
python glue_bert_gaussian.py --task stsb --output_dir ./STSB -seed 42 --training_set_pct 100
python glue_bert_gaussian.py --task wnli --output_dir ./WNLI -seed 42 --training_set_pct 100
```

## Thanks

If our research aids yours, please do not forget to cite us! :)
