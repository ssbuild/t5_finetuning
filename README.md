## 安装

- pip install -U deep_training >= 0.1.0
- 当前文档版本pypi 0.1.0

## 更新详情

- [deep_training](https://github.com/ssbuild/deep_training)

## 深度学习常规任务例子

- [deep_training-pytorch-example](https://github.com/ssbuild/deep_training-pytorch-example)
- [deep_training-tf-example](https://github.com/ssbuild/deep_training-tf-example)


## clue-chat finetuning 

    预训练模型下载
- [ChatYuan-large-v1](https://huggingface.co/ClueAI/ChatYuan-large-v1)
- [ChatYuan-large-v2](https://huggingface.co/ClueAI/ChatYuan-large-v2)
    
    

## 数据示例
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

## 生成训练record

    python data_utils.py
    
    注:
    num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)


## 训练

    python train.py

## 推理
    
    infer.py   只能推理微调模型 (权重参考train.py 最后部分转换) 
    infer_chatyuan.py  推理微调模型和lora模型
