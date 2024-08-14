from FlagEmbedding import FlagReranker
from FlagEmbedding import BGEM3FlagModel
import pandas as pd
from tqdm import tqdm


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


def entity_alignment_e(bodypart):
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

if __name__ == '__main__':
    df2 = pd.read_csv('./data/bodyparts.csv', encoding='utf-8')
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
            res = entity_alignment_e(bodypart)
            df.iloc[index, 1] = res
    df.to_excel('./data/SignKG-e.xlsx', index=False)
