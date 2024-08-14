import openpyxl  
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
            
def m3e(list,lists):
    model = SentenceTransformer('m3e-base')

    #Sentences are encoded by calling model.encode()
    embeddings1 = model.encode(list)
    for i in lists:
        embeddings2 = model.encode(i)
        pc = 0
        for vec1, vec2 in zip(embeddings1, embeddings2):
            sim = cosine_similarity([vec1], [vec2])[0][0]
            if sim > 0.9:
                pc+=1
                continue
        if pc==4:
            return 0
        else:
            continue
    return 1

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

    worsee = 0
    # 遍历'sign'中的每个子列表  
    for sign_sublist in tqdm(sign_sublists):  
        # worsec = bgem3(sign_sublist,label_sublists)
        worsec = m3e(sign_sublist,label_sublists)
        worsee += worsec
    # return missing_count   
    return worsee 

def calculate_recall_number(dict):
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

    
    label_sublists = data_dict['label.xlsx']
    sign_sublists = data_dict['SignKG-e.xlsx']
    worsen = acc_n(label_sublists,sign_sublists)
    accn = 1-(worsen/len(data_dict['label.xlsx']))

    worsee = count_missing_elements(data_dict)

    r_acc = 1 - worsee / len(data_dict['./data/label.xlsx'])
    print(f"Relation Recall: {rrecall:.4f}")
    print(f"Relation Accuracy: {r_acc:.4f}")

