# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py


import copy
import json
import os
import random
import typing

import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments
from deep_training.utils.func import is_chinese_char
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from tqdm import tqdm
from transformers import T5Tokenizer, HfArgumentParser

train_info_args = {
    'devices': 1,
    'data_backend': 'record',
    'model_type': 't5',
    # 预训练模型路径 , 从0训练，则置空
    'model_name_or_path': '/data/nlp/pre_models/torch/t5/ChatYuan-large-v1',
    'tokenizer_name': '/data/nlp/pre_models/torch/t5/ChatYuan-large-v1',
    'config_name': '/data/nlp/pre_models/torch/t5/ChatYuan-large-v1/config.json',
    'convert_onnx': False, # 转换onnx模型
    'do_train': True, 
    'train_file':  [ '/data/nlp/nlp_train_data/chatyuan/finetune_train_examples.json'],
    'max_epochs': 3,
    'train_batch_size': 6,
    'eval_batch_size': 2,
    'test_batch_size': 2,
    'optimizer': 'adamw',
    'learning_rate': 5e-5,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length': 512,
    'max_target_length': 100  # 预测最大长度
}

data_conf = {
    'stride': 50, #滑动窗口 ， 数据多则相应增大，否则减小
    'count_per_group': 1, #大规模训练可以提高 ,#chatyuan 应该是1
}
def preprocess(text):
  text = text.replace("\n", "\\n").replace("\t", "\\t")
  return text

def postprocess(text):
  return text.replace("\\n", "\n").replace("\\t", "\t")


class NN_DataHelper(DataHelper):
    index = 1

    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1

        tokenizer: T5Tokenizer
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer

        stride = data_conf['stride']
        COUNT_PER_GROUP = data_conf['count_per_group']

        sub_list = data
        input_ids = []

        for idx, paragraphs in enumerate(sub_list):
            if COUNT_PER_GROUP > 1:
                text = paragraphs + '<extra_id_0>'
            else:
                text = paragraphs
            o = tokenizer.encode_plus(text=text, truncation=True,
                                      return_attention_mask=False,
                                      return_token_type_ids=False)
            if len(o['input_ids']) <= 3:
                continue
            input_ids += o['input_ids'][:-1]
            if idx != len(sub_list) - 1:
                input_ids += [tokenizer.eos_token_id]

        pos = 0
        ds = []
        while pos < len(input_ids):
            input_ids_ = input_ids[pos: pos + max_seq_length - 2] + [tokenizer.eos_token_id]
            pos += stride

            if len(input_ids_) <= 5:
                continue
            seqlen = np.asarray(len(input_ids_), dtype=np.int32)
            pad_len = max_seq_length - seqlen
            input_ids_ = np.asarray(input_ids_, dtype=np.int32)
            if pad_len:
                pad_val = tokenizer.pad_token_id
                input_ids_ = np.pad(input_ids_, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            d = {
                'input_ids': input_ids_,
                'seqlen': seqlen
            }
            ds.append(d)
        if self.index < 3:
            print(ds[0])
        return ds


    #{"id": 0, "paragraph": ["用户：写一个诗歌，关于冬天", "小元：冬夜寂静冷，", "云在天边飘，", "冰封白雪上， ", "寒冷像一场雪。", " ", "雪花融化成冰，", "像那雪花飘洒，", "在寒冷的冬天，", "感受春天的喜悦。", " 冬日里，", "风雪渐消，", "一片寂静，", "把快乐和温暖带回家。"]}
    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        COUNT_PER_GROUP = data_conf['count_per_group']

        sub = []
        for file in files:
            with open(file,mode='r',encoding='utf-8',newline='\n') as f:
                lines = f.readlines()

            for i,line in enumerate(lines):
                jd = json.loads(line)
                if not jd:
                    continue
                paragraph = jd['paragraph']
                if i < 10:
                    print(paragraph)
                texts = ''
                for text in paragraph:
                    texts += preprocess(text + '\n')
                sub.append(texts)
                if len(sub) >= COUNT_PER_GROUP:
                    D.append(copy.deepcopy(sub))
                    sub.clear()
        if sub:
            D.append(copy.deepcopy(sub))
        return D

    def collate_fn(self, batch):
        self.tokenizer: T5Tokenizer
        o = {}
        for i, b in enumerate(batch):
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])

        seqlens = o.pop('seqlen')
        max_len = torch.max(seqlens).numpy().tolist()

        bs = len(batch)
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id


        input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(size=(bs, max_len), dtype=torch.long)
        decoder_input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
        decoder_attention_mask = torch.zeros(size=(bs, max_len), dtype=torch.long)
        labels = torch.full((bs, max_len), -100, dtype=torch.long)

        a_maxlen, b_maxlen = 0, 0
        raw_input_ids = o.pop('input_ids')

        for (seqlen, ids, a_ids, a_mask, b_ids, b_mask, label) in zip(seqlens, raw_input_ids, input_ids, attention_mask,
                                                                      decoder_input_ids, decoder_attention_mask,
                                                                      labels):
            seqlen = seqlen.squeeze(-1).numpy().tolist()
            s = np.random.randint(1, seqlen - 1, dtype=np.int32).tolist()
            a_ids[:s] = ids[:s]
            a_ids[s] = eos_token_id
            a_mask[:s + 1] = 1

            b_ids[:seqlen - s] = ids[s:seqlen]
            b_mask[:seqlen - s] = 1
            label[:seqlen - s-1] = b_ids[1:seqlen - s]
            a_maxlen = max(a_maxlen, s + 1)
            b_maxlen = max(b_maxlen, seqlen - s)

        o['input_ids'] = input_ids[:, :a_maxlen]
        o['attention_mask'] = attention_mask[:, :a_maxlen]
        o['decoder_input_ids'] = decoder_input_ids[:, :b_maxlen]
        o['decoder_attention_mask'] = decoder_attention_mask[:, :b_maxlen]
        o['labels'] = labels[:, :b_maxlen]
        return o


if __name__ == '__main__':

    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config()
    config.decoder_start_token_id = tokenizer.cls_token_id
    # 缓存数据集
    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file, shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, shuffle=False,mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file, shuffle=False,mode='test')


    def shuffle_records(record_filenames, outfile, compression_type='GZIP'):
        print('shuffle_records record...')
        options = RECORD.TFRecordOptions(compression_type=compression_type)
        dataset_reader = Loader.RandomDataset(record_filenames, options=options, with_share_memory=True)
        data_size = len(dataset_reader)
        all_example = []
        for i in tqdm(range(data_size), desc='load records'):
            serialized = dataset_reader[i]
            all_example.append(serialized)
        dataset_reader.close()

        shuffle_idx = list(range(data_size))
        random.shuffle(shuffle_idx)
        writer = WriterObject(outfile, options=options)
        for i in tqdm(shuffle_idx, desc='shuffle record'):
            example = all_example[i]
            writer.write(example)
        writer.close()

    # 对每个record 再次打乱
    for filename in dataHelper.train_files:
        shuffle_records(filename, filename)
