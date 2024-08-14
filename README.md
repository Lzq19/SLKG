# SLKG
The Automated Construction Method of the Chinese Sign Language Knowledge Graph
## Installation
```bash
$ git clone https://github.com/Lzq19/SLKG.git
$ cd SLKG
$ conda install --yes --file requirements.txt
```
## Fine-tunning
Follow the glm fine-tuning method. The training file is `P-train.json`, the testing file is `P-test.json`.
```bash
https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/README.md
```
## Structure
Placement of checkpoint according to the following structure.
```bash
├── SLKG/
│   └── data/
│   └── checkpoint/
```
## Eval
```bash
$ python eval.py
```
## Web Demo
```bash
$ python show.py
```