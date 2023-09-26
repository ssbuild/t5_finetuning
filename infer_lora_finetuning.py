# @Time    : 2023/3/19 18:15
# @Author  : tk
# @FileName: infer
import os

import torch
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser
from aigc_zoo.model_zoo.t5.llm_model import MyTransformer,PetlArguments
from data_utils import train_info_args, postprocess, NN_DataHelper


def generate_text(base_model,text,device = torch.device('cuda:0'),max_length=128):
    input_text = "用户：" + text + "\n小元："

    o = tokenizer.encode_plus(input_text, truncation=True, max_length=512, return_attention_mask=False,return_token_type_ids=False)
    input_ids= [o['input_ids']]
    input_ids = torch.tensor(input_ids, dtype=torch.int32,device=device)

    logits = base_model.generate(input_ids=input_ids, max_length=max_length, bos_token_id=config.decoder_start_token_id,
                                 pad_token_id=config.pad_token_id,
                                 eos_token_id=config.eos_token_id)


    out_text = tokenizer.decode(logits[0], skip_special_tokens=True)
    out_text = postprocess(out_text)
    return out_text

if __name__ == '__main__':

    parser = HfArgumentParser((ModelArguments, ))
    (model_args,) = parser.parse_dict(train_info_args,allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, config, _,_= dataHelper.load_tokenizer_and_config()

    ckpt_dir = './best_ckpt/last'
    lora_args = PetlArguments.from_pretrained(ckpt_dir)
    assert lora_args.inference_mode == True

    new_num_tokens = config.vocab_size
    if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
        config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config,model_args=model_args,lora_args=lora_args,
                             torch_dtype=config.torch_dtype,
                             new_num_tokens=new_num_tokens,
                             )


    # 加载lora权重
    pl_model.load_sft_weight(ckpt_dir)

    enable_merge_weight = False
    if enable_merge_weight:
        # 合并lora 权重 保存
        pl_model.save_sft_weight(os.path.join(ckpt_dir, 'pytorch_model_merge.bin'), merge_lora_weight=True)
    else:
        model = pl_model.get_llm_model()
        model.eval().half().cuda()


        text= "写一个诗歌，关于冬天"
        output = generate_text(model,text)
        print('input',text)
        print('output',output)