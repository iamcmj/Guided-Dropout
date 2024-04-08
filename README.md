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

## Computing Fisher Information Matrices
The fisher_matrix_{model}.py scripts compute the empirical Fisher (eqn 3 in the paper) and take in the randomization seed and training set percentage as arguments:

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
The python scripts will build the required subdirectories current the current directory for you.
