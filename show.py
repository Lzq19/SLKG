from transformers import AutoModel,AutoTokenizer
import pandas as pd
import gradio as gr
from PyQt5.QtWidgets import QFileDialog, QApplication
import sys
from test import model_act


def data_change(data_):
    if data_ == "上传数据":
        file_path = select_file()
        return pd.read_excel(file_path)
    
    
def select_file():
    app = QApplication(sys.argv)
    file_path, _ = QFileDialog.getOpenFileName()
    app.quit()
    return file_path


def get_response(prompt):
    response = model.chat(tokenizer, prompt, history=[])
    res = response[0].replace('，', ',').replace('；', ';')
    if res[-1] == ';':
        res = res[:-1]
    lines = [line.strip() for line in res.split(';')]
    data = [line.split(',') for line in lines]
    return data

def create_kg(prompt):
        data = get_response(prompt)
        while not all(len(sublist) == 4 for sublist in data):
            prompt += '输出的每个;前由四部分组成，请修正'
            data = get_response(prompt)
        
        df = pd.DataFrame(data, columns=["手语词","肢体部位","关系动作","时序属性"])
        return df  
            
        

def function1(input1):
    prompt = rf"""我将交给你一个“任务”，请根据“示例”，从“给定文本”中提取“知识图谱三元组”，并按动作发生顺序标注。
    #示例：
    <示例1>给定文本: 看: 一手食、中指分开，指尖朝前，掌心向下，从眼部向前一指。(可根据实际表示看的动作)。 
    输出: 看, 右手食指, 分开, 1;看, 右手中指, 分开, 1;看, 右手食指指尖, 朝前, 2;看, 右手中指指尖, 朝前, 2;看, 右手掌心, 向下, 3;看, 右手食指, 从眼部向前一指, 4;看, 右手中指, 从眼部向前一指, 4。
    <示例2>给定文本: 馄饨: 右手食指在左手掌心上一抹，如抹上肉馅，左手随即握拳，仿包馄饨的动作。输出: 馄饨, 右手食指, 在左手掌心上一抹, 1;馄饨, 左手, 握拳, 2。
    #任务
    //给定文本：{input1}。
    输出：
    """

    df = create_kg(prompt)
    yield df



        
def function2(face2_input1,face2_input2):
    querys = []
    for _, row in face2_input2.iterrows():
        row_str = ':'.join(row.values.astype(str))
        querys.append(row_str)

    querys2 = []
    for query in querys:
        prompt = rf"""我将交给你一个“任务”，请根据“示例”，从“给定文本”中提取“知识图谱三元组”，并按动作发生顺序标注。
        #示例：
        <示例1>给定文本: 看: 一手食、中指分开，指尖朝前，掌心向下，从眼部向前一指。(可根据实际表示看的动作)。 
        输出: 看, 右手食指, 分开, 1;看, 右手中指, 分开, 1;看, 右手食指指尖, 朝前, 2;看, 右手中指指尖, 朝前, 2;看, 右手掌心, 向下, 3;看, 右手食指, 从眼部向前一指, 4;看, 右手中指, 从眼部向前一指, 4。
        <示例2>给定文本: 馄饨: 右手食指在左手掌心上一抹，如抹上肉馅，左手随即握拳，仿包馄饨的动作。输出: 馄饨, 右手食指, 在左手掌心上一抹, 1;馄饨, 左手, 握拳, 2。
        #任务
        //给定文本：{query}。
        输出：
        """
        querys2.append(prompt)

    all_res = []
    for t_query in querys2:
        df = create_kg(t_query)
        all_res.append(df)
    res = pd.concat(all_res, ignore_index=True, axis=0)
    yield res
    




if __name__ == "__main__":
    model,tokenizer = model_act()

    input1 =gr.Textbox(label="手语词：手语动作描述",lines=1)
    output2 = gr.DataFrame(pd.DataFrame(columns=["手语词","肢体部位","关系动作","时序属性"]),label="手语知识图谱")

    face2_input1 = gr.Radio(choices=["上传数据"],label="批量数据")
    face2_input2 =gr.DataFrame(pd.DataFrame(columns=["手语词","手语动作描述"]))
    face2_onput2 = gr.DataFrame(pd.DataFrame(columns=["手语词","肢体部位","关系动作","时序属性"]),label="手语知识图谱")

    iface1 = gr.Interface(function1,[input1],[output2],submit_btn="图谱生成",allow_flagging="never",clear_btn="清除")
    iface2 = gr.Interface(function2, [face2_input1,face2_input2],[face2_onput2],submit_btn="图谱生成",allow_flagging="never",clear_btn=gr.Button("清除",visible=False))


    tabbed_interface = gr.TabbedInterface([iface1, iface2], ["手动输入", "批量输入"], title="手语知识图谱自动化构建")

    with tabbed_interface as tt:
        face2_input1.change(data_change, inputs=face2_input1, outputs=face2_input2)
        tt.launch(server_name='0.0.0.0', share=False)
