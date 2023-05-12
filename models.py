# @Time    : 2023/4/2 23:14
# @Author  : tk
# @FileName: models
from deep_training.nlp.models.lora import LoraArguments, LoraModel
from deep_training.nlp.models.transformer import TransformerForSeq2SeqLM
from transformers import T5ForConditionalGeneration


class MyTransformer(TransformerForSeq2SeqLM, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraArguments = kwargs.pop('lora_args')
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        if lora_args.with_lora:
            model = LoraModel(self.backbone,lora_args)
            print('*' * 30)
            model.print_trainable_parameters()
            self.set_model(model,copy_attr=False)

    def get_llm_model(self) -> T5ForConditionalGeneration:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        return self.backbone.model