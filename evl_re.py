import openpyxl  
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

        
def m3e(list,lists):
    model = SentenceTransformer('moka-ai/m3e-base')
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

    for row in sheet.iter_rows(values_only=True):  
        row_data = [str(cell).strip() for cell in row]   
        all_rows.append(row_data)  
      
    return all_rows
  

def count_missing_elements(data_dict):  
    label_sublists = data_dict['./data/label.xlsx']
    sign_sublists = data_dict['./data/SignKG-e.xlsx']

    worsee = 0
    for sign_sublist in tqdm(sign_sublists):  
        worsec = m3e(sign_sublist,label_sublists)
        worsee += worsec
    return worsee 

def calculate_recall_number(dict):
    label = dict['./data/label.xlsx']
    predict = dict['./data/SignKG-e.xlsx']
    pc = len(label)-len(predict)
    return pc

if __name__=='__main__':
    file_paths = ['./data/label.xlsx', './data/SignKG-e.xlsx']  
    data_dict = {}  
    
    for file_path in file_paths:  
        data_dict[file_path] = process_excel(file_path)    

    recall_w = calculate_recall_number(data_dict)
    rrecall = 1-(recall_w/(len(data_dict['./data/label.xlsx'])))

    worsee = count_missing_elements(data_dict)

    r_acc = 1 - worsee / len(data_dict['./data/label.xlsx'])
    print(f"Relation Recall: {rrecall:.4f}")
    print(f"Relation Accuracy: {r_acc:.4f}")

