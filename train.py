# -*- coding: utf-8 -*-
#reference: https://github.com/clue-ai/PromptCLUE/blob/main/Fine_tuning_PyTorch.ipynb
import logging

import torch
import transformers
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.nlp.models.lora import LoraModel, LoraArguments
from deep_training.nlp.models.transformer import TransformerForSeq2SeqLM
from deep_training.utils.trainer import SimpleModelCheckpoint
from pytorch_lightning import Trainer
from transformers import HfArgumentParser, T5Tokenizer, T5Config

from data_utils import NN_DataHelper, train_info_args, preprocess, postprocess


class MyTransformer(TransformerForSeq2SeqLM, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraArguments = kwargs.pop('lora_args')
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        if lora_args.with_lora:
            model = LoraModel(self.backbone,lora_args)
            print('*' * 30)
            model.print_trainable_parameters()
            self.set_model(model,copy_attr=False)

class MySimpleModelCheckpoint(SimpleModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(MySimpleModelCheckpoint, self).__init__(*args, **kwargs)
        lora_args: LoraArguments = self.external_kwargs['lora_args']
        if lora_args.with_lora:
            self.weight_file = './best_ckpt'
            self.last_weight_file = './last_ckpt'


    def load_model_from_ckpt(self):
        model_args = self.external_kwargs['model_args']
        training_args = self.external_kwargs['training_args']
        lora_args = LoraArguments.from_pretrained(self.last_weight_file)
        pl_module = MyTransformer(lora_args=lora_args,
                              config=config,
                              model_args=model_args,
                              training_args=training_args)


        pl_module.backbone.from_pretrained(pl_module.backbone.model,self.last_weight_file)
        return pl_module

    @staticmethod
    def generate_text(pl_module: MyTransformer, prefix, tokenizer, max_target_length, device=0):
        device = torch.device('cuda:{}'.format(device))
        config = pl_module.config

        model: transformers.T5ForConditionalGeneration
        if pl_module.lora_args.with_lora: # lora 从权重加载模型
            model = pl_module.backbone.model.model
            model.to(device)
            model.eval()
        else:
            model = pl_module.backbone.model

        # 简易测试生成
        o = tokenizer.encode_plus(prefix, truncation=True, max_length=512, return_attention_mask=False,return_token_type_ids=False)
        input_ids= [o['input_ids']]
        input_ids = torch.tensor(input_ids, dtype=torch.int32,device=device)

        logits = model.generate(input_ids,max_length=max_target_length,bos_token_id=config.decoder_start_token_id,
                                pad_token_id=config.pad_token_id,
                                eos_token_id=config.eos_token_id)


        out_text = tokenizer.decode(logits[0], skip_special_tokens=True)
        out_text = postprocess(out_text)
        return out_text

    def on_save_model(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        lora_args : LoraArguments =  self.external_kwargs['lora_args']
        # 保存权重
        if not lora_args.with_lora:
            super(MySimpleModelCheckpoint, self).on_save_model(trainer, pl_module)
        else:
            monitor_candidates = self._monitor_candidates(trainer)
            monitor_candidates.update(self.on_get_metric(trainer, pl_module))
            val = monitor_candidates.get(self.monitor, None)

            #保存loss最小权重
            if self.update_best(val):
                logging.info('epoch {} ,step {} , save best {}, {}\n'.format(monitor_candidates['epoch'],
                                                                             monitor_candidates['step'],
                                                                             self.best[self.monitor],
                                                                             self.weight_file))
                pl_module.backbone.save_pretrained(self.weight_file)
            #保存最新权重
            pl_module.backbone.save_pretrained(self.last_weight_file)
            # 从最新权重加载模型
            pl_module = self.load_model_from_ckpt()


        prefixs = [
            "帮我写一个请假条，我因为新冠不舒服，需要请假3天，请领导批准",
            "你能干什么",
            "用英文写一封道歉的邮件，表达因为物流延误，不能如期到达，我们可以赔偿贵公司所有损失",
            "写一个文章，题目是未来城市",
            "写一个诗歌，关于冬天",
            "从南京到上海的路线",
            "学前教育专业岗位实习中，在学生方面会存在问题，请提出改进措施。800字",
            "根据标题生成文章：标题：屈臣氏里的化妆品到底怎么样？正文：化妆品，要讲究科学运用，合理搭配。屈臣氏起码是正品连锁店。请继续后面的文字。",
            "帮我对比几款GPU，列出详细参数对比，并且给出最终结论",
        ]

        device = trainer.global_rank
        self.tokenizer: T5Tokenizer
        tokenizer = self.tokenizer
        data_args = self.data_args


        print('*' * 30,'generate_text...')
        for text in prefixs:
            input_text = "用户：" + text + "\n小元："
            input_text = preprocess(input_text)
            output = MySimpleModelCheckpoint.generate_text(pl_module, input_text, tokenizer,data_args.max_target_length, device=device)

            print('input', text)
            print('output', output)
            print()



if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments,LoraArguments))
    model_args, training_args, data_args,lora_args = parser.parse_dict(train_info_args)
    # 保存最小loss模型
    checkpoint_callback = MySimpleModelCheckpoint(monitor="loss",
                                                  every_n_epochs = 1,
                                                  every_n_train_steps=2000 // training_args.gradient_accumulation_steps,
                                                  #模型参数
                                                  model_args=model_args,
                                                  training_args=training_args,
                                                  lora_args=lora_args,
                                                  )
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,
        enable_progress_bar=True,
        default_root_dir=data_args.output_dir,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        num_sanity_val_steps=0,
        strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',
    )

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config()

    # 额外参数
    checkpoint_callback.tokenizer = tokenizer
    checkpoint_callback.data_args = data_args

    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False,shuffle=True,mode='train',num_process_worker=0)
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file,mode='test')


    model = MyTransformer(lora_args=lora_args,config=config,model_args=model_args, training_args=training_args)

    if not data_args.convert_onnx:
        train_datasets = dataHelper.load_distributed_random_sampler(
            dataHelper.train_files,
            with_load_memory=True,
            collate_fn=dataHelper.collate_fn,
            batch_size=training_args.train_batch_size,
            drop_last=True,  # 多卡建议扔掉
            num_processes=trainer.world_size, process_index=trainer.global_rank)

        if train_datasets is not None:
            trainer.fit(model, train_dataloaders=train_datasets)
        # else:
        #     eval_datasets = dataHelper.load_sequential_sampler(dataHelper.eval_files,batch_size=training_args.eval_batch_size,collate_fn=dataHelper.collate_fn)
        #     test_datasets = dataHelper.load_sequential_sampler(dataHelper.test_files,batch_size=training_args.test_batch_size,collate_fn=dataHelper.collate_fn)
        #     if eval_datasets is not None:
        #         trainer.validate(model, dataloaders=eval_datasets, ckpt_path='./best.pt')
        #
        #     if test_datasets is not None:
        #         trainer.test(model, dataloaders=test_datasets, ckpt_path='best.pt')
    else:
        # 加载权重
        if not lora_args.with_lora:
            model = MyTransformer.load_from_checkpoint('./best.pt',
                                                       lora_args=lora_args,
                                                       config=config,
                                                       model_args=model_args,
                                                       training_args=training_args)
            model_: transformers.T5ForConditionalGeneration
            model_ = model.backbone.model
            #保存权重, 可选上传至huggingface
            tokenizer: T5Tokenizer
            config: T5Config
            tokenizer.save_pretrained('chatyuan_finetuning')
            config.save_pretrained('chatyuan_finetuning')
            model_.save_pretrained('chatyuan_finetuning', push_to_hub = False,max_shard_size= "10GB")


            #转换onnx 模型
            input_sample = (
                ("input_ids", torch.ones(size=(1, 128), dtype=torch.int32)),
                ("attention_mask", torch.ones(size=(1, 128), dtype=torch.int32)),
                ("decoder_input_ids", torch.ones(size=(1, 128), dtype=torch.int32)),
                ("decoder_attention_mask", torch.ones(size=(1, 128), dtype=torch.int32)),
            )
            input_names = ("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
            output_names = ("pred_ids",)
            dynamic_axes = None or {"input_ids": [0, 1], "attention_mask": [0, 1],
                                    "decoder_input_ids": [0, 1], "decoder_attention_mask": [0, 1],
                                    "pred_ids": [0, 1]}
            model.convert_to_onnx('./best.onnx',
                                  input_sample=input_sample,
                                  input_names=input_names,
                                  output_names=output_names,
                                  dynamic_axes=dynamic_axes)
        else:
            #加载权重
            lora_args = LoraArguments.from_pretrained('./best_ckpt')
            pl_module = MyTransformer(lora_args=lora_args,
                                      config=config,
                                      model_args=model_args,
                                      training_args=training_args)
            #二次加载权重
            pl_module.backbone.from_pretrained(pl_module.backbone.model,'./best_ckpt')

            model_: transformers.T5ForConditionalGeneration
            model_ = pl_module.backbone.model.model
