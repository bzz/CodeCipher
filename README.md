## Introduction
Code for "CODECIPHER: LEARNING TO OBFUSCATE SOURCE CODE AGAINST LLMS". CodeCipher is a novel method that protects code privacy while preserving the original responses from large language models (LLMs). It achieves this by transforming the LLM's embedding matrix, such that each row corresponds to a different word in the original matrix. This transformation creates a token-to-token confusion mapping, effectively obfuscating the source code. The new embedding matrix is optimized by minimizing a task-specific loss function.

## Repository Structure
```
├── data # training data
├── data_loader_gen.py # data loader for code generation
├── data_loader.py # data loader for code summarization
├── data_loader_translate.py # data loader for code translation
├── human-eval
├── modeling_llama.py # model for LLM
├── README.md
├── requirements.txt
├── train_matrix_iter_gen.py # code for running code completion task
├── train_matrix_iter_summary.py # code for running code summarization task
├── train_matrix_iter_translate.py # code for running code translation task
└── utils # utility functions
```

## Dataset
The trainning data for code summary can be downloaded from [here](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text). Others is in the data folder.

## Environment
```
pip install requirements.txt
pip install -e human-eval
```
## Run
To run the code, you can execute the following commands. A detailed list of hyperparameters can be found in Appendix C of the paper.

```
python train_matrix_iter_gen.py # code for code completion
python train_matrix_iter_summary.py # code for summarization
python train_matrix_iter_translate.py # code for translation
```

