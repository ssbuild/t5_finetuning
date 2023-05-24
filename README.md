## update information
   - [deep_training](https://github.com/ssbuild/deep_training)

```text
    05-24 lora v2
```

## install
  - pip install -i https://pypi.org/simple -U deep_training==0.1.8 transformers>=4.28 deepspeed




## weight

- [ChatYuan-large-v1](https://huggingface.co/ClueAI/ChatYuan-large-v1)
- [ChatYuan-large-v2](https://huggingface.co/ClueAI/ChatYuan-large-v2)
    
    

## data sample
    单条数据示例
    {
        "id": 0, "paragraph": [
            #一轮会话
            {
                "q": "从南京到上海的路线",
                "a": [
                    "你好，南京到上海的路线如下：",
                    "1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。",
                    "2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。",
                    "3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"
                ]
            }
            #二轮....
        ]
    }


# 使用方法
    默认不使用滑动窗口
    data_conf = {
        'stride': 0,
        #滑动窗口 , 数据多则相应增大，否则减小 ,stride <=0 则禁用滑动窗口
    }

## generate record

    python data_utils.py
    
    注:
    num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)


## finetuning

    python train.py

## infer
    
    infer.py 推理官方模型权重
    infer_finetuning.py  推理微调模型权重
    infer_lora_finetuning.py 和lora 微调模型权重

## 友情链接

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [tf-task-example](https://github.com/ssbuild/tf-task-example)
- [chatmoss_finetuning](https://github.com/ssbuild/chatmoss_finetuning)
- [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
- [chatyuan_finetuning](https://github.com/ssbuild/chatyuan_finetuning)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [rlhf_llm](https://github.com/ssbuild/rlhf_llm)
- [rlhf_chatglm](https://github.com/ssbuild/rlhf_chatglm)
- [rlhf_chatyuan](https://github.com/ssbuild/rlhf_chatyuan)

## 
    纯粹而干净的代码