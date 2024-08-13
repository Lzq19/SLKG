from FlagEmbedding import FlagReranker
from FlagEmbedding import BGEM3FlagModel
import pandas as pd
from tqdm import tqdm


# 只使用了reranker
def enity_alignment_r(bodypart):
    # 读取JSON文件
    df = pd.read_csv('/home/aii/lzq/SignKG/data/bodyparts.csv', encoding='utf-8')

    # 提取每一行的最后一个元素
    bodyparts_dict = df.iloc[:, -1].tolist()
    score_list = []
    # reranker = FlagReranker('/home/aii/lzq/SignKG/thirdparty/bge/bge-reranker-large', use_fp16=True)
    for bodypart_ in bodyparts_dict:
        score = reranker.compute_score([bodypart,bodypart_])
        score_list.append(score)
    # 只用分数
    # max_score = max(score_list)
    # max_index = score_list.index(max_score)
    # return bodyparts_dict[max_index]

    # 只用分数有缺陷，优化
    indexs = sorted(range(len(score_list)), key=lambda i: score_list[i], reverse=True)[:2]
    top2 = [bodyparts_dict[i] for i in indexs]
    if len(top2[0]) == len(top2[1]):
        return top2[0]
    else:
        for top in top2:
            if len(bodypart) < len(top):
                return top


def enity_alignment_e(bodypart):
    model = BGEM3FlagModel('/data/SignKG/thirdparty/bge/bge-m3', use_fp16=True)
    # 读取JSON文件
    df = pd.read_csv('/data/SignKG/data/bodyparts.csv', encoding='utf-8')

    # 提取每一行的最后一个元素
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

if __name__ == '__main__':
    df2 = pd.read_csv('/data/SignKG/data/bodyparts.csv', encoding='utf-8')
    bodyparts_dict2 = df2.iloc[:, -1].tolist()

    reranker = FlagReranker('/data/SignKG/thirdparty/bge/bge-reranker-large', use_fp16=True)

    file_path = '/data/SignKG/outputs/SignKG-no-e.xlsx'
    df = pd.read_excel(file_path)
    bodyparts_data = df.iloc[:, 1].tolist()
    for index, bodypart in tqdm(enumerate(bodyparts_data)):
        bodypart=bodypart.replace(" ", "")  
        if bodypart in bodyparts_dict2:
            continue
        else:
            res = enity_alignment_e(bodypart)
            df.iloc[index, 1] = res
    df.to_excel('/data/SignKG/outputs/SignKG-e.xlsx', index=False)
    # res_e = enity_alignment_e('右手掌')
    # print(res)
    # print(res_e)
