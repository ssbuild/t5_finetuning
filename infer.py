# @Time    : 2023/3/19 18:25
# @Author  : tk
# @FileName: infer_chatyuan


# 使用
import torch
from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration


tokenizer = T5Tokenizer.from_pretrained("./ChatYuan-large-v1")
model = T5ForConditionalGeneration.from_pretrained("./ChatYuan-large-v1")
# 修改colab笔记本设置为gpu，推理更快
device = torch.device('cuda')
model.to(device)


def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text


def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t")


def answer(text, sample=True, top_p=1, temperature=0.7):
    '''sample：是否抽样。生成任务，可以设置为True;
    top_p：0-1之间，生成的内容越多样'''
    text = preprocess(text)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(
        device)
    if not sample:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512,
                             num_beams=1, length_penalty=0.6)
    else:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512,
                             do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])




input_text0 = "帮我写一个请假条，我因为新冠不舒服，需要请假3天，请领导批准"
input_text1 = "你能干什么"
input_text2 = "写一封英文商务邮件给英国客户，表达因为物流延误，不能如期到达，我们可以赔偿贵公司所有损失"
input_text3 = "写一个文章，题目是未来城市"
input_text4 = "写一个诗歌，关于冬天"
input_text5 = "从南京到上海的路线"
input_text6 = "学前教育专业岗位实习中，在学生方面会存在问题，请提出改进措施。800字"
input_text7 = "根据标题生成文章：标题：屈臣氏里的化妆品到底怎么样？正文：化妆品，要讲究科学运用，合理搭配。屈臣氏起码是正品连锁店。请继续后面的文字。"
input_text8 = "帮我对比几款GPU，列出详细参数对比，并且给出最终结论"
input_list = [input_text0, input_text1, input_text2, input_text3, input_text4, input_text5, input_text6, input_text7, input_text8]
for i, input_text in enumerate(input_list):
  input_text = "用户：" + input_text + "\n小元："
  print(f"示例{i}".center(50, "="))
  output_text = answer(input_text)
  print(f"{input_text}{output_text}")