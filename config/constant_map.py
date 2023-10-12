# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps

from aigc_zoo.constants.define import (TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING)

__all__ = [
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "train_model_config"
]

train_info_models = {
    'ChatYuan-large-v2': {
        'model_type': 't5',
        'model_name_or_path': '/data/nlp/pre_models/torch/t5/ChatYuan-large-v2',
        'tokenizer_name': '/data/nlp/pre_models/torch/t5/ChatYuan-large-v2',
        'config_name': '/data/nlp/pre_models/torch/t5/ChatYuan-large-v2/config.json',
    },
    'ChatYuan-large-v1': {
        'model_type': 't5',
        'model_name_or_path': '/data/nlp/pre_models/torch/t5/ChatYuan-large-v1',
        'tokenizer_name': '/data/nlp/pre_models/torch/t5/ChatYuan-large-v1',
        'config_name': '/data/nlp/pre_models/torch/t5/ChatYuan-large-v1/config.json',

    },
    'PromptCLUE-base-v1-5': {
        'model_type': 't5',
        'model_name_or_path': '/data/nlp/pre_models/torch/t5/PromptCLUE-base-v1-5',
        'tokenizer_name': '/data/nlp/pre_models/torch/t5/PromptCLUE-base-v1-5',
        'config_name': '/data/nlp/pre_models/torch/t5/PromptCLUE-base-v1-5/config.json',
    },

}


# 按需修改
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING

