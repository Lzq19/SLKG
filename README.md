# SLKG
The Automated Construction Method of the Chinese Sign Language Knowledge Graph
## Installation
```bash
$ git clone https://github.com/Lzq19/SLKG.git
$ cd SLKG
$ conda install --yes --file requirements.txt
```
## Models
Download the model according to the following command.
```bash
$ git clone https://huggingface.co/THUDM/chatglm-6b
$ git clone https://huggingface.co/BAAI/bge-m3
$ git clone https://huggingface.co/BAAI/bge-reranker-large
```
## Structure
Placement of models and checkpoint according to the following structure.
```bash
├── SLKG/
│   └── data/
│   └── checkpoint/
│   └── thirdparty/
│       ├── glm
│            └── chatglm-6b
│       ├── bge
│            └── bge-reranker-large
│            └── bge-m3
```
## Fine-tunning
Follow the glm fine-tuning method. The training file is `P-train.json`, the testing file is `P-test.json`.
```bash
https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/README.md
```
## Test
```bash
$ python test.py
```
## Web Demo
```bash
$ python show.py
```