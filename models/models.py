# @Time    : 2023/4/2 23:14
# @Author  : tk
# @FileName: models
from deep_training.nlp.models.lora.v2 import LoraArguments, LoraConfig, LoraModel
from deep_training.nlp.models.transformer import TransformerForSeq2SeqLM
from transformers import T5ForConditionalGeneration


class MyTransformerLM(TransformerForSeq2SeqLM):
    def __init__(self, *args, **kwargs):
        # 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
        load_in_8bit = kwargs.get('load_in_8bit', False)
        if not load_in_8bit:
            kwargs.pop("device_map", None)
        super(MyTransformerLM, self).__init__(*args, **kwargs)


    def enable_input_require_grads(self):
        # setattr(self.model, 'model_parallel', True)
        # setattr(self.model, 'is_parallelizable', True)
        self.model.enable_input_require_grads()

