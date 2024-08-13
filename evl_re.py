import openpyxl  
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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


# 定义处理Excel文件的函数  
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

    # for row in sheet.iter_rows(values_only=True):  
    #     # 只获取前三列数据  
    #     row_data = [cell.strip() for cell in row[:3] if cell is not None]  
    #     # 确保row_data至少有3个元素，不足则用None补齐  
    #     row_data.extend([None] * (3 - len(row_data)))  
    #     all_rows.append(row_data)  
      
    # return all_rows 
  
# 文件路径列表，替换为你的文件路径  
file_paths = ['label.xlsx', 'SignKG-e.xlsx']  
  
# 初始化一个空字典来保存每个文件的数据  
data_dict = {}  
  
# 对每个文件进行处理，并将结果保存在字典中  
for file_path in file_paths:  
    data_dict[file_path] = process_excel(file_path)  

# 计算'sign'列表中不在'label'列表中的元素数量  
def count_missing_elements(data_dict):  
    label_sublists = data_dict['label.xlsx']  # 将'label'列表转换为集合  
    sign_sublists = data_dict['SignKG-e.xlsx']  # 'sign'列表 

    worse = 0
    missing_count = 0
    # 遍历'sign'中的每个子列表  
    for sign_sublist in tqdm(sign_sublists):  
        worsec = m3e(sign_sublist,label_sublists)
        worse += worsec
        # # 检查该子列表是否在'label'中的任何子列表中  
        # if sign_sublist  in label_sublists:  
        #     continue
        # missing_count += 1  
      
    # return missing_count   
    return worse 
  
missing_count = count_missing_elements(data_dict)  
print(missing_count)  # 输出'sign'列表中不在'label'列表中的元素数量