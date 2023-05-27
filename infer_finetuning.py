# @Time    : 2023/3/19 18:15
# @Author  : tk
# @FileName: infer
import os

import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser

from models import MyTransformer,LoraArguments
from data_utils import train_info_args, postprocess, NN_DataHelper,get_deepspeed_config


def generate_text(base_model,text,device = torch.device('cuda:0'),max_length=128):
    input_text = "用户：" + text + "\n小元："

    o = tokenizer.encode_plus(input_text, truncation=True, max_length=512, return_attention_mask=False,return_token_type_ids=False)
    input_ids= [o['input_ids']]
    input_ids = torch.tensor(input_ids, dtype=torch.int32,device=device)

    logits = base_model.generate(input_ids,max_length=max_length,bos_token_id=config.decoder_start_token_id,
                            pad_token_id=config.pad_token_id,
                            eos_token_id=config.eos_token_id)


    out_text = tokenizer.decode(logits[0], skip_special_tokens=True)
    out_text = postprocess(out_text)
    return out_text


deep_config = get_deepspeed_config()

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_dict(train_info_args, allow_extra_keys=True)



    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer, config, _,_= dataHelper.load_tokenizer_and_config()

    pl_model = MyTransformer(config=config, model_args=model_args)

    ###################### 注意 选最新权重
    # 选择最新的权重 ， 根据时间排序 选最新的

    if deep_config is None:
        train_weight = './best_ckpt/last-v3.ckpt'
        assert os.path.exists(train_weight)

    else:
        # 建议直接使用转换脚本命令 支持 deepspeed stage 0,1,2,3， 生成 ./best_ckpt/last.ckpt/best.pt 权重文件
        # cd best_ckpt/last.ckpt
        # python zero_to_fp32.py . best.pt
        train_weight = './best_ckpt/last.ckpt/best.pt'


    #加载权重
    pl_model.load_sft_weight('./best.pt')

    model = pl_model.get_llm_model()
    model.eval().cuda()

    text= "写一个诗歌，关于冬天"
    output = generate_text(model,text)
    print('input',text)
    print('output',output)