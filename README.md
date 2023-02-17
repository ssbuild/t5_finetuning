## 安装

- pip install -U deep_training >= 0.0.15
- 当前文档版本pypi 0.0.15

## 更新详情

- [deep_training](https://github.com/ssbuild/deep_training)

## 深度学习常规任务例子

- [deep_training-examples](https://github.com/ssbuild/deep_training-examples)


## clue-chat finetuning 

    预训练模型下载 https://huggingface.co/ClueAI/ChatYuan-large-v1

## 数据示例

{"id": 0, "paragraph": ["用户：写一个诗歌，关于冬天", "小元：冬夜寂静冷，", "云在天边飘，", "冰封白雪上， ", "寒冷像一场雪。", " ", "雪花融化成冰，", "像那雪花飘洒，", "在寒冷的冬天，", "感受春天的喜悦。", " 冬日里，", "风雪渐消，", "一片寂静，", "把快乐和温暖带回家。"]}



# 使用方法

## 生成训练record

python data_utils.py

## 训练

python task_chat_t5.py
