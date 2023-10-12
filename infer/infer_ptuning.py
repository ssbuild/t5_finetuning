# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer_ptuning
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser,AutoConfig

from data_utils import train_info_args, NN_DataHelper, postprocess
from aigc_zoo.model_zoo.t5.llm_model import MyTransformer,PetlArguments,PromptArguments

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
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)



    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs={"torch_dtype": torch.float16})

    ckpt_dir = './best_ckpt/last'
    config = AutoConfig.from_pretrained(ckpt_dir)
    prompt_args = PromptArguments.from_pretrained(ckpt_dir)

    assert prompt_args.inference_mode == True

    new_num_tokens = config.vocab_size
    if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
        config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args, prompt_args=prompt_args,
                             torch_dtype=config.torch_dtype,
                             new_num_tokens=new_num_tokens,
                             )
    # 加载sft权重
    pl_model.load_sft_weight(ckpt_dir)

    pl_model.eval().half().cuda()

    model = pl_model.get_llm_model()

    #基础模型精度
    model.base_model_torch_dtype = torch.half

    text = "写一个诗歌，关于冬天"
    output = generate_text(model, text)
    print('input', text)
    print('output', output)