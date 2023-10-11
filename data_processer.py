# @Time    : 2023/3/25 18:36
# @Author  : tk
import copy
from enum import Enum
import numpy as np
from transformers import PreTrainedTokenizer



class DataStrategy(Enum):
    tunction = 1
    slidding = 2





def build_template_chatyuan(query, answer = None,prefix=None, history=None):
    prompt = prefix or ''
    if history is not None:
        for q,a in history:
            prompt += "用户：{}小元：".format(q,a)
    prompt += "用户：{}小元：".format(query)
    if answer is not None:
        prompt += answer
    return prompt

def build_template_default(query, answer = None,prefix=None, history=None):
    prompt = prefix or ''
    if history is not None:
        for q,a in history:
            prompt += "User: {}\nAssistant:{}".format(q,a)
    prompt += "User: {}\nAssistant:".format(query)
    if answer is not None:
        prompt += answer
    return prompt

def build_template_tiger(query,answer = None,prefix=None, history=None):
    prompt = prefix or ''
    tok_ins = "\n\n### Instruction:\n"
    tok_res = "\n\n### Response:\n"
    if history is not None:
        for q,a in history:
            prompt += "{}{}{}{}".format(tok_ins,q,tok_res,a)

    prompt += "{}{}{}".format(tok_ins, query, tok_res)
    if answer is not None:
        prompt += answer
    return prompt


#切换模板
build_template = build_template_chatyuan

class TokenTunction:

    @classmethod
    def final(cls,a_ids,b_ids,max_seq_length):
        seqlen = len(a_ids)
        decoder_seqlen = len(b_ids)

        pad_len = max_seq_length - seqlen
        if pad_len > 0:
            a_ids += [0] * pad_len

        pad_len = max_seq_length - decoder_seqlen
        if pad_len > 0:
            b_ids += [0] * pad_len

        labels = copy.deepcopy(b_ids[1:]) + [-100]
        labels = np.asarray(labels, dtype=np.int64)
        labels[decoder_seqlen:] = -100

        d = {
            'input_ids': np.asarray(a_ids, dtype=np.int32),
            'attention_mask': np.asarray([1] * seqlen, dtype=np.int32),
            'seqlen': np.asarray(seqlen, dtype=np.int32),
            'decoder_input_ids': np.asarray(b_ids, dtype=np.int32),
            'decoder_attention_mask': np.asarray([1] * decoder_seqlen, dtype=np.int32),
            'decoder_seqlen': np.asarray(decoder_seqlen, dtype=np.int32),
            'labels': np.asarray(labels, dtype=np.int64)
        }
        return d
    @classmethod
    def process(cls, tokenizer: PreTrainedTokenizer, config, sup, max_seq_length, examples):
        ds = []
        prefix, examples = examples
        for sid, (q, a) in enumerate(examples):
            a_ids = tokenizer.encode(text=build_template(q, history=examples[:sid]), add_special_tokens=False)
            b_ids = tokenizer.encode(text=a,add_special_tokens=False)
            while len(a_ids) > max_seq_length :
                a_ids.pop(0)
            while len(b_ids) > max_seq_length - 2:
                b_ids.pop(-1)
            b_ids = [config.decoder_start_token_id] + b_ids + [config.eos_token_id]
            ds.append(cls.final(a_ids,b_ids,max_seq_length))
        return ds


class TokenSlidding:
    @classmethod
    def process(cls, tokenizer: PreTrainedTokenizer,config,stride,sup, max_seq_length, examples):
        ds = []
        prefix,examples = examples
        for sid, (q, a) in enumerate(examples):
            a_ids = tokenizer.encode(text=build_template(q, history=examples[:sid]), add_special_tokens=False)
            b_ids = tokenizer.encode(text=a) + [config.eos_token_id]

            input_ids_all = a_ids + b_ids

            pos = 0
            while pos < len(input_ids_all):
                input_ids = [config.bos_token_id] + input_ids_all[pos: pos + max_seq_length - 1]
                pos += stride

                ds.append({
                'input_ids': np.asarray(input_ids,dtype=np.int32),
                'seqlen': np.asarray(len(input_ids),dtype=np.int32)
            })
        return ds


