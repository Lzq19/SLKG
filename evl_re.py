import openpyxl  
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

def bgem3(lista,listb):
    model = BGEM3FlagModel('./thirdparty/bge/bge-m3', use_fp16=True) 
    pc = 0
    embeddings1 = model.encode(lista)
    embeddings2 = model.encode(listb)
    for vec1, vec2 in zip(embeddings1['dense_vecs'], embeddings2['dense_vecs']):
        sim = vec1 @ vec2.T
        if sim > 0.9:
            pc+=1
            continue
    if pc==4:
        return 0
    else:
        return 1
    
def acc_n(lista,listb):
    worse = 0
    for vec1, vec2 in zip(lista, listb):
        if vec1==vec2 or vec1 in listb:
            continue
        else:
            worse += 1
    return worse 
            

def process_excel(file_path):  
    workbook = openpyxl.load_workbook(file_path)  
    sheet = workbook.active  
    all_rows = []  
    # 读取所有列
    for row in sheet.iter_rows(values_only=True):  
        # 读取所有列并删除空格  
        row_data = [str(cell).strip() for cell in row]   
        all_rows.append(row_data)  
      
    return all_rows
  

def count_missing_elements(data_dict):  
    label_sublists = data_dict['./data/label.xlsx']
    sign_sublists = data_dict['./data/SignKG-e.xlsx']

    worse = 0
    for predict,gt in tqdm(zip(sign_sublists,label_sublists)):
        worsec = bgem3(predict,gt)
        worsen = acc_n(predict,gt)
        worse += worsec
 
    return worse 

def RRC(dict):
    label = dict['label.xlsx']
    predict = dict['SignKG-e.xlsx']
    pc = len(label)-len(predict)
    return pc

if __name__=='__main__':
    file_paths = ['./data/label.xlsx', './data/SignKG-e.xlsx']  

    data_dict = {}  
    
    for file_path in file_paths:  
        data_dict[file_path] = process_excel(file_path)    

    recall_w = RRC(data_dict)
    rrecall = 1-(recall_w/(len(data_dict['./data/label.xlsx'])))

    
    missing_count = count_missing_elements(data_dict)

    r_acc_e = (len(data_dict['./data/label.xlsx'])-missing_count)/len(data_dict['./data/label.xlsx'])
    print(f"Relation Recall: {rrecall:.4f}")
    print(f"Relation Accuracy (Normal): {acc_n:.4f}")
    print(f"Relation Accuracy (Enhanced): {r_acc_e:.4f}")

