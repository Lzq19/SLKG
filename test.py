from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import os
import json
from tqdm import tqdm
from entity_alignment import entity_alignment_e
import copy
import pandas as pd
from openpyxl import load_workbook, Workbook
from evl_re import process_excel,count_missing_elements,calculate_recall_number
from evl_entity import sp
from F1 import calculate_f1


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


if __name__ == "__main__":
    # predict
    file_path = './data/P-test.json'
    wb2 = Workbook() 
    ws2 = wb2.active

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            json_data = json.loads(line)
            content = json_data.get('content','')
            summary = json_data.get('summary','')
            p = summary.replace('，', ',').replace('；', ';')
            if p[-1] == ';':
                p = p[:-1]
            lines2 = [line2.strip() for line2 in p.split(';')]
            data2 = [line2.split(',') for line2 in lines2]
            for d2 in data2:
                ws2.append(d2)
            data.append(content)
            wb2.save("./data/label.xlsx")

    wb3 = Workbook() 
    ws3 = wb3.active

    model,tokenizer = model_act()
    for query in tqdm(data):
        res_predict = model.chat(tokenizer, query, history=[])
        res = res_predict[0].replace('，', ',').replace('；', ';')
        if res[-1] == ';':
            res = res[:-1]
        lines = [line.strip() for line in res.split(';')]
        data = [line.split(',') for line in lines]

        data_alignment = copy.deepcopy(data)
        df = pd.read_csv('./data/bodyparts.csv', encoding='utf-8')
        bodyparts_dict = df.iloc[:, -1].tolist()
        for de in data_alignment:
            flag = de[1]
            if de[1] in bodyparts_dict:
                continue
            else:
                de[1] = entity_alignment_e(de[1])
            ws3.append(de)

    wb3.save("./data/SignKG-e.xlsx")


    # eval
    label_book = load_workbook(rf'./data/label.xlsx')  
    Sign_e_book = load_workbook(rf'./data/SignKG-e.xlsx')
    
    worksheet_l = label_book.active  
    worksheet_e = Sign_e_book.active 
   
    label = []
    Sign_e = []
 
    for row in worksheet_l.iter_rows(min_row=1, values_only=True):  
        row_as_list = list(row)  
        label.append(row_as_list)  

    for row in worksheet_e.iter_rows(min_row=1, values_only=True):  
        row_as_list = list(row)  
        Sign_e.append(row_as_list) 

    lbg = sp(label)
    Seg = sp(Sign_e)

    right_e = fn = 0
    for key in Seg:
        for t1 in Seg[key]:
            for sublist in lbg[key]:
                body_predict = t1[1].replace(" ", "")
                lab = sublist[1].replace(" ", "")
                if body_predict == lab:
                    right_e+=1
                    break

    entity_recall = 1-(fn/(len(label)-fn))
    entity_acc_alignment = right_e/len(label)

    print(f"Entity Recall: {entity_recall:.4f}")
    print(f"Entity Accuracy: {entity_acc_alignment:.4f}")


    file_paths = ['./data/label.xlsx', './data/SignKG-e.xlsx']  
    data_dict = {}  
    
    for file_path in file_paths:  
        data_dict[file_path] = process_excel(file_path)    

    relation_recall_number = calculate_recall_number(data_dict)
    relation_recall = 1-(relation_recall_number/(len(data_dict['./data/label.xlsx'])))

    relation_worse_number = count_missing_elements(data_dict)
    relation_acc = 1 - relation_worse_number / len(data_dict['./data/label.xlsx'])

    print(f"Relation Recall: {relation_recall:.4f}")
    print(f"Relation Accuracy: {relation_acc:.4f}")


    entity_f1_score_alignment = calculate_f1(entity_acc_alignment, entity_recall)  
    relation_f1_score_no_alignment = calculate_f1(relation_acc, relation_recall)  

    print(f"Entity F1 Score (Enhanced): {entity_f1_score_alignment:.4f}")
    print(f"Relation F1 Score: {relation_f1_score_no_alignment:.4f}")
