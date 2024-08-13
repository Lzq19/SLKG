from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import os
import json
from openpyxl import load_workbook
from FlagEmbedding import FlagReranker
from FlagEmbedding import BGEM3FlagModel
import pandas as pd
from tqdm import tqdm  


def model_act():
    ptuning_checkpoint = fr'./checkpoint'
    model_path = './thirdparty/glm/chatglm-6b'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=512)
    model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(ptuning_checkpoint, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model = model.quantize(4)
    model = model.half().cuda()
    model.transformer.prefix_encoder.float()
    model = model.eval()
    return model,tokenizer


def enity_alignment_r(bodypart):
    df = pd.read_csv('./data/bodyparts.csv', encoding='utf-8')

    bodyparts_dict = df.iloc[:, -1].tolist()
    score_list = []
    for bodypart_ in bodyparts_dict:
        score = reranker.compute_score([bodypart,bodypart_])
        score_list.append(score)

    indexs = sorted(range(len(score_list)), key=lambda i: score_list[i], reverse=True)[:2]
    top2 = [bodyparts_dict[i] for i in indexs]
    if len(top2[0]) == len(top2[1]):
        return top2[0]
    else:
        for top in top2:
            if len(bodypart) < len(top):
                return top

def enity_alignment_e(bodypart):
    model = BGEM3FlagModel('./thirdparty/bge/bge-m3', use_fp16=True)
    df = pd.read_csv('./data/bodyparts.csv', encoding='utf-8')

    bodyparts_dict = df.iloc[:, -1].tolist()
    similarity_dict = []
    embeddings_1 = model.encode(bodypart, return_dense=True, return_sparse=True, return_colbert_vecs=True)
    for bodypart_ in bodyparts_dict:
        embeddings_2 = model.encode(bodypart_, return_dense=True, return_sparse=True, return_colbert_vecs=True)
        similarity = model.colbert_score(embeddings_1['colbert_vecs'], embeddings_2['colbert_vecs'])
        similarity_dict.append(similarity)

    max_similarity = max(similarity_dict)
    max_index = similarity_dict.index(max_similarity)
    return bodyparts_dict[max_index]

def process_row(row):
    return [item.upper() if index==1 else item for index, item in enumerate(row)]


def sp(label):
    # 使用字典来根据第一个字符串元素分组  
    label_groups = {}
    for sublist in label:
        key = sublist[0]  # 第一个字符串元素作为键  
        if key not in label_groups:
            label_groups[key] = []
        label_groups[key].append(sublist)  # 将整个子列表添加到对应的组中  
    return label_groups

def calculate_f1(precision, recall):  
    if precision == 0 or recall == 0:  
        return 0  
    else:  
        f1 = 2 * (precision * recall) / (precision + recall)  
        return f1  


def output_to_excel(filename,out_file_name):
    ##################将大模型输出的知识图谱文件写到excel中#################
    # 假设您的文本数据存储在名为data.txt的文件中

    # 读取文件并解析每行数据
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data_p = []
            line = line.replace('\n', '')
            parts = line.strip().split(';')
            parts = [element for element in parts if element != '']

            for part in parts:
                units = part.strip().split(',')
                if len(units) == 4:
                    data_p.append(units)
                else:
                    print(units)
                    data_p = []
                    break
            else:
                for d in data_p:
                    data.append(d)

                        # 转换为pandas DataFrame
    df = pd.DataFrame(data, columns=['词', '身体部位', '动作', '时间序号'])

    # 将DataFrame写入Excel文件，如果需要，可以更改文件名
    df.to_excel(out_file_name, index=False)


if __name__ == "__main__":
    
    file_path = './data/P-test.json'
    out_path = './data/P-test_output.txt'

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            json_data = json.loads(line)
            content = json_data.get('content','')
            data.append(content)

    with open(out_path, 'w', encoding='utf-8') as o:
        model,tokenizer = model_act()
        for query in data:
            res = model.chat(tokenizer, query, history=[])
            o.write(res[0]+'\n')
    
    output_to_excel(out_path,'./data/SignKG-no-e.xlsx')


    df2 = pd.read_csv('./bodyparts.csv', encoding='utf-8')
    bodyparts_dict2 = df2.iloc[:, -1].tolist()

    reranker = FlagReranker('./thirdparty/bge/bge-reranker-large', use_fp16=True)

    file_path = './data/SignKG-no-e.xlsx'
    df = pd.read_excel(file_path)
    bodyparts_data = df.iloc[:, 1].tolist()
    for index, bodypart in tqdm(enumerate(bodyparts_data)):
        bodypart=bodypart.replace(" ", "")  
        if bodypart in bodyparts_dict2:
            continue
        else:
            res = enity_alignment_e(bodypart)
            df.iloc[index, 1] = res
    df.to_excel('./data/SignKG-e.xlsx', index=False)


# 加载Excel工作簿  
labelbook = load_workbook(rf'./data/label.xlsx')  
Signbook = load_workbook(rf'./data/SignKG-no-e.xlsx')
Signebook = load_workbook(rf'./data/SignKG-e.xlsx')
# 选择第一个工作表（你也可以通过名字选择工作表）  
worksheet_l = labelbook.active  
worksheet_n = Signbook.active 
worksheet_e = Signebook.active 
# 初始化一个空列表，用于存放每一行的列表  
label = []
Sign_n = []
Sign_e = []
# 遍历工作表中的每一行（从第二行开始，因为第一行通常是标题行）  
for row in worksheet_l.iter_rows(min_row=1, values_only=True):  
    # 将每一行的值转换为一个列表，并添加到rows_as_lists中  
    row_as_list = list(row)  
    label.append(row_as_list)  

for row in worksheet_n.iter_rows(min_row=1, values_only=True):  
    # 将每一行的值转换为一个列表，并添加到rows_as_lists中  
    row_as_list = list(row)  
    Sign_n.append(row_as_list) 

for row in worksheet_e.iter_rows(min_row=1, values_only=True):  
    # 将每一行的值转换为一个列表，并添加到rows_as_lists中  
    row_as_list = list(row)  
    Sign_e.append(row_as_list) 

lbg = sp(label)
Sng = sp(Sign_n)
Seg = sp(Sign_e)

right_n=right_e=fn=0
for key in Sng:
    for t in Sng[key]:
        t = list(filter(None.__ne__, t))
        if len(t) != 4:
            fn+=1
        for sublist in lbg[key]:
            bodyp=t[1].replace(" ", "")
            lab = sublist[1].replace(" ", "")
            if bodyp == lab:
                right_n+=1
                break

for key in Seg:
    for t1 in Seg[key]:
        for sublist in lbg[key]:
            bodyp=t1[1].replace(" ", "")
            lab = sublist[1].replace(" ", "")
            if bodyp == lab:
                right_e+=1
                break

recall = 1-(fn/(len(label)-fn))
acc_n = right_n/len(label)
acc_e = right_e/len(label)
print(f"Entity Recall: {recall:.4f}")
print(f"Entity Accuracy (Normal): {acc_n:.4f}")
print(f"Entity Accuracy (Enhanced): {acc_e:.4f}")









# 计算F1分数  
f1_score_n = calculate_f1(acc_n, recall)  
f1_score_e = calculate_f1(acc_e, recall)  
  
# 输出F1分数  
print(f"Entity F1 Score (Normal): {f1_score_n:.4f}")
print(f"Entity F1 Score (Enhanced): {f1_score_e:.4f}")
