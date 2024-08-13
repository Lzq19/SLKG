from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import os
import json


def model_act():
    STEP=3000
    CHECKPOINT='adgen-chatglm-6b-pt-512-2e-2'
    ptuning_checkpoint = fr'./checkpoint/{CHECKPOINT}/checkpoint-{STEP}'
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
    # model = AutoModel.from_pretrained(ptuning_checkpoint, trust_remote_code=True)
    model = model.quantize(4)
    model = model.half().cuda()
    model.transformer.prefix_encoder.float()
    model = model.eval()
    return model,tokenizer


if __name__ == "__main__":
    
    file_path = 'data/P-test.json'
    out_path = 'data/P-test_output.txt'

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
            # print(res)
            o.write(res[0]+'\n')