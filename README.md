# SLKG
The Automated Construction Method of the Chinese Sign Language Knowledge Graph
## Installation
```bash
$ git clone https://github.com/yourusername/SLKG.git
$ cd SLKG
$ conda install --yes --file requirements.txt
```

```bash
$ git lfs install
```
GLM models
```bash
$ git clone https://huggingface.co/THUDM/chatglm-6b
$ git clone https://huggingface.co/THUDM/chatglm2-6b
$ git clone https://huggingface.co/THUDM/chatglm3-6b
```
Embedding & Reranker models
```bash
$ git clone https://huggingface.co/BAAI/bge-m3
$ git clone https://huggingface.co/BAAI/bge-reranker-large
```
## Usage
```bash
$ python run.py
```
## Gradio Demo
```bash
$ python show.py
```