# -*- coding: utf-8 -*-
# @Time    : 2023/5/24 15:53
import json
import os

from config.constant_map import train_info_models

train_model_config = train_info_models['ChatYuan-large-v2']

#如果显卡支持int8 可以开启
global_args = {
    "load_in_8bit": False, # lora 如果显卡支持int8 可以开启
    "load_in_4bit": False,

    #load_in_4bit 量化配置
    "quantization_config": None,
    "config_merge": {
    },
    "num_layers": -1,  # 是否使用骨干网络的全部层数 ， -1 表示全层, 否则只用只用N层
}



train_info_args = {
    'devices': 1,
    'data_backend': 'record',  #one of record lmdb arrow_stream arrow_file,parquet, 超大数据集可以使用 lmdb , 注 lmdb 存储空间比record大
    # 预训练模型路径
    **train_model_config,

    'convert_onnx': False, # 转换onnx模型
    'do_train': True,
    'convert_file': True, # train_file是否需要制作record , 如果已经制作好，可以不需要原语料文件，train_file 为制作好的record 文件list
    'train_file':  [ './data/finetune_train_examples.json'],
    'max_epochs': 3,
    'max_steps': -1,
    'train_batch_size': 4,
    'eval_batch_size': 2,
    'test_batch_size': 2,
    'learning_rate': 5e-5, # lora 调大学习率 1e-3
    'optimizer': 'lion',
    # one of [lamb,adamw_hf,adamw,adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_apex_fused,adafactor,adamw_anyprecision,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,lion_8bit,lion_32bit,paged_adamw_32bit,paged_adamw_8bit,paged_lion_32bit,paged_lion_8bit]

    'scheduler_type': 'CAWR',
    # one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau, cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau]
    'scheduler':{'T_mult': 1, 'rewarm_epoch_num': 2, 'verbose': False},


    # 'scheduler_type': 'linear',# one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau
    # 'scheduler': None,

    # 切换scheduler类型
    # 'scheduler_type': 'WarmupCosine',
    # 'scheduler': None,

    # 'scheduler_type': 'ReduceLROnPlateau',
    # 'scheduler': None,

    # 'scheduler_type': 'Step',
    # 'scheduler':{ 'decay_rate': 0.999,'decay_steps': 100,'verbose': True},

    # 'scheduler_type': 'CAWR',
    # 'scheduler':{'T_mult': 1, 'rewarm_epoch_num': 2, 'verbose': False},

    # 'scheduler_type': 'CAL',
    # 'scheduler': {'rewarm_epoch_num': 2,'verbose': True},
    'optimizer_betas': (0.9, 0.999),
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length': 512,
    'max_target_length': 100,  # 预测最大长度

}

