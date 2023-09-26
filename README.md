## update information
   - [deep_training](https://github.com/ssbuild/deep_training)

```text
    09-26 support transformers trainer
    08-02 增加 muti lora infer 例子, 手动升级 aigc_zoo , pip install -U git+https://github.com/ssbuild/aigc_zoo.git --force-reinstall --no-deps
	06-13 support resize_token_embeddings
    06-01 支持lora deepspeed 训练，0.1.9 和 0.1.10合并
    05-27 add qlora transformers>=4.30
    05-24 升级 lora
```

## install
  - pip install -U -r requirements.txt
  - 如果无法安装， 可以切换官方源 pip install -i https://pypi.org/simple -U -r requirements.txt


## weight

- [ChatYuan-large-v1](https://huggingface.co/ClueAI/ChatYuan-large-v1)
- [ChatYuan-large-v2](https://huggingface.co/ClueAI/ChatYuan-large-v2)
    
    
## data
[open data](https://github.com/ssbuild/open_data)
```text
p prefix  optional
q question optional
a answer   must

```
```json
 {
    "id": 0, 
    "p": "我是qwen训练的模型",
    "paragraph": [
        {
           "q": "你好",
           "a": "我是机器人，有什么可以帮助你的？"
        },
         {
             "q": "从南京到上海的路线",
             "a":  "你好，南京到上海的路线如下：1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"
         }
     ]
 }

```

或者

```json
 {
    "id": 0,
    "conversations": [
      {
        "from": "system",
        "value": "我是qwen训练的模型"
      },
      {
        "from": "user",
        "value": "你好"
      },
      {
        "from": "assistant",
        "value": "我是机器人，有什么可以帮助你的？"
      },
      {
        "from": "user",
        "value": "从南京到上海的路线"
      },
      {
        "from": "assistant",
        "value": "你好，南京到上海的路线如下：1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"
      }
     ]
 }
```


# 使用方法
    默认不使用滑动窗口
    data_conf = {
        'stride': 0,
        #滑动窗口 , 数据多则相应增大，否则减小 ,stride <=0 则禁用滑动窗口
    }





## infer
    # infer_finetuning.py 推理微调模型
    # infer_lora_finetuning.py 推理微调模型
    # infer_ptuning.py 推理p-tuning-v2微调模型
     python infer_finetuning.py



## training
```text
    #制作数据
    python data_utils.py
    注: num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)
    
    #训练
    python train.py
```
   
## 训练参数
[训练参数](args.MD)

## 友情链接

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [tf-task-example](https://github.com/ssbuild/tf-task-example)
- [chatmoss_finetuning](https://github.com/ssbuild/chatmoss_finetuning)
- [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
- [t5_finetuning](https://github.com/ssbuild/t5_finetuning)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [llm_rlhf](https://github.com/ssbuild/llm_rlhf)
- [chatglm_rlhf](https://github.com/ssbuild/chatglm_rlhf)
- [t5_rlhf](https://github.com/ssbuild/t5_rlhf)
- [rwkv_finetuning](https://github.com/ssbuild/rwkv_finetuning)
- [baichuan_finetuning](https://github.com/ssbuild/baichuan_finetuning)

## 
    纯粹而干净的代码