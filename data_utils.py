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
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments, TrainingArgumentsHF, \
    TrainingArgumentsCL
from aigc_zoo.model_zoo.t5.llm_model import PetlArguments,PromptArguments
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from tqdm import tqdm
from transformers import T5Tokenizer, HfArgumentParser, T5Config
from config import *
from data_processer import DataStrategy, TokenTunction, TokenSlidding

data_conf = {
    'strategy': DataStrategy.tunction,  # 数据策略选项
    DataStrategy.tunction: {
        'ensure_answer_min_length': 1,
        'sup': True, # 是否监督模式
    },

    DataStrategy.slidding: {
        'stride': int(train_info_args['max_seq_length'] / 3 * 2),
        'sup': True, # 是否监督模式
    }

}

def preprocess(text):
  return text.replace("\n", "\\n").replace("\t", "\\t")

def postprocess(text):
  return text.replace("\\n", "\n").replace("\\t", "\t")


class NN_DataHelper(DataHelper):
    index = 1
    def __init__(self, *args,**kwargs):
        super(NN_DataHelper, self).__init__(*args,**kwargs)

        strategy = data_conf['strategy']
        if strategy == DataStrategy.tunction:
            self.collate_fn = self.collate_fn_none_stride
        else:
            #滑动窗口模式
            self.collate_fn = self.collate_fn_stride


    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1

        tokenizer: T5Tokenizer
        config: T5Config
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer # noqa
        config = self.config # noqa
        examples = data

        strategy = data_conf['strategy']
        if strategy == DataStrategy.tunction:
            ds = TokenTunction.process(tokenizer, config=config, max_seq_length=max_seq_length, examples=examples,
                                       **data_conf[strategy])
        elif strategy == DataStrategy.slidding:
            ds = TokenSlidding.process(tokenizer, config=config, max_seq_length=max_seq_length, examples=examples,
                                       **data_conf[strategy])

        else:
            raise ValueError('Invalid strategy', strategy)
        if not ds:
            return None

        if self.index < 3:
            print(ds[0])
        return ds



    def _get_paragraph(self, lines):
        D = []
        for line_id, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            paragraph = jd['paragraph']
            if line_id < 10:
                print(paragraph)

            prefix = jd.get('p', '')
            paragraph = [(preprocess(session['q']),
                          preprocess('\n'.join(session['a'])) if isinstance(session['a'], list) else preprocess(
                              session['a']))
                         for session in paragraph]
            sub = []
            for (q, a) in paragraph:
                assert len(a), ValueError('answer cannot empty')
                sub.append((preprocess(q), preprocess(a)))
            D.append((prefix, copy.deepcopy(sub)))
        return D

    def _get_messages(self, lines):
        D = []
        for line_id, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            conversations = jd['conversations']
            if line_id < 10:
                print(conversations)

            paragraph = []
            prefix = ''
            pair = [None, None]
            for m in conversations:
                if m["from"] == 'user':
                    pair[0] = preprocess(m["value"])
                elif m["from"] == 'assistant':
                    pair[1] = preprocess(m["value"])
                elif m["from"] == 'system':
                    prefix = preprocess(m["value"])
                if pair[0] is not None and pair[1] is not None:
                    paragraph.append(tuple(pair))
                    pair[0], pair[1] = None, None

            sub = []
            for (q, a) in paragraph:
                assert len(a), ValueError('answer cannot empty')
                sub.append((preprocess(q), preprocess(a)))
            D.append((prefix, copy.deepcopy(sub)))
        return D

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()
            is_new = False
            if len(lines) > 0:
                is_new = 'conversations' in json.loads(lines[0])
            if is_new:
                D.extend(self._get_messages(lines))
            else:
                D.extend(self._get_paragraph(lines))
        return D



    def collate_fn_stride(self, batch):
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
        decoder_start_token_id = self.config.decoder_start_token_id


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

            if ids[0] != decoder_start_token_id:
                b_len = seqlen - s + 1
                b_ids[0] = decoder_start_token_id
                b_ids[1:b_len] = ids[s:seqlen]
                b_mask[:b_len] = 1
                label[:b_len- 1] = b_ids[1:b_len]
            else:
                b_len = seqlen - s
                b_ids[:b_len] = ids[s:seqlen]
                b_mask[:b_len] = 1
                label[:b_len - 1] = b_ids[1:b_len]

            a_maxlen = max(a_maxlen, s + 1)
            b_maxlen = max(b_maxlen, b_len)

        o['input_ids'] = input_ids[:, :a_maxlen].long()
        o['attention_mask'] = attention_mask[:, :a_maxlen].long()
        o['decoder_input_ids'] = decoder_input_ids[:, :b_maxlen].long()
        o['decoder_attention_mask'] = decoder_attention_mask[:, :b_maxlen].long()
        o['labels'] = labels[:, :b_maxlen].long()
        return o

    def collate_fn_none_stride(self, batch):
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

        seqlen = torch.sum(o.pop('seqlen'))
        decoder_seqlen = torch.sum(o.pop('decoder_seqlen'))

        o['input_ids'] = o['input_ids'][:,:seqlen].long()
        o['attention_mask'] = o['attention_mask'][:,:seqlen].long()
        o['decoder_input_ids'] = o['decoder_input_ids'][:,:decoder_seqlen].long()
        o['decoder_attention_mask'] = o['decoder_attention_mask'][:,:decoder_seqlen].long()
        o['labels'] = o['labels'][:,:decoder_seqlen].long()
        return o

    def make_dataset_all(self):
        data_args = self.data_args

        #schema for arrow parquet
        schema = {
            "input_ids": "int32_list",
            "attention_mask": "int32_list",
            "seqlen": "int32_list",
            "decoder_input_ids": "int32_list",
            "decoder_attention_mask": "int32_list",
            "decoder_seqlen": "int32_list",
            "labels": "int32_list",
        }
        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train',
                                              schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval',schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test',schema=schema)

if __name__ == '__main__':
    if global_args[ "trainer_backend" ] == "hf":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsHF, DataArguments, PetlArguments, PromptArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )
    elif global_args[ "trainer_backend" ] == "pl":
        parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments, PromptArguments))
        model_args, training_args, data_args, _, _ = parser.parse_dict(train_info_args)
    else:
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsCL, DataArguments, PetlArguments, PromptArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config()

    # 缓存数据集
    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    dataHelper.make_dataset_all()


    # def shuffle_records(record_filenames, outfile, compression_type='GZIP'):
    #     print('shuffle_records record...')
    #     options = RECORD.TFRecordOptions(compression_type=compression_type)
    #     dataset_reader = Loader.RandomDataset(record_filenames, options=options, with_share_memory=True)
    #     data_size = len(dataset_reader)
    #     all_example = []
    #     for i in tqdm(range(data_size), desc='load records'):
    #         serialized = dataset_reader[i]
    #         all_example.append(serialized)
    #     dataset_reader.close()
    #
    #     shuffle_idx = list(range(data_size))
    #     random.shuffle(shuffle_idx)
    #     writer = WriterObject(outfile, options=options)
    #     for i in tqdm(shuffle_idx, desc='shuffle record'):
    #         example = all_example[i]
    #         writer.write(example)
    #     writer.close()
    #
    # # 对每个record 再次打乱
    # for filename in dataHelper.train_files:
    #     shuffle_records(filename, filename)
