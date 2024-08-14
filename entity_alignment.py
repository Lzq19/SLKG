from FlagEmbedding import FlagReranker
from FlagEmbedding import BGEM3FlagModel
import pandas as pd
from tqdm import tqdm


def entity_alignment_e(bodypart):
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
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

    reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

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
