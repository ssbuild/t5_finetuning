# @Time    : 2023/3/19 18:15
# @Author  : tk
# @FileName: infer
import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser
from models import MyTransformer,LoraArguments
from data_utils import train_info_args, postprocess, NN_DataHelper


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

if __name__ == '__main__':

    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args)



    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _,_= dataHelper.load_tokenizer_and_config()

    # 加载权重
    if not lora_args.with_lora:
        #非lora模式，可以按照 chatyuan官方 直接模式
        pl_model = MyTransformer.load_from_checkpoint('./best.pt',
                                                   lora_args=lora_args,
                                                   config=config,
                                                   model_args=model_args,
                                                   training_args=training_args)
        base_model = pl_model.get_llm_model()


    else:
        # 加载权重
        lora_args = LoraArguments.from_pretrained('./best_ckpt')
        assert lora_args.inference_mode
        pl_model = MyTransformer(lora_args=lora_args,
                                  config=config,
                                  model_args=model_args,
                                  training_args=training_args)
        # 二次加载权重
        pl_model.backbone.from_pretrained(pl_model.backbone.model, './best_ckpt')

        base_model = pl_model.get_llm_model()

    base_model.eval()
    base_model.cuda()



    text= "帮我写一个请假条，我因为新冠不舒服，需要请假3天，请领导批准"
    output = generate_text(base_model,text)
    print('input',text)
    print('output',output)