# -*- coding: utf-8 -*-
#reference: https://github.com/clue-ai/PromptCLUE/blob/main/Fine_tuning_PyTorch.ipynb
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.trainer.pl.modelcheckpoint import ModelCheckpointEx
from lightning import Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy
from transformers import HfArgumentParser
from data_utils import NN_DataHelper, train_info_args,get_deepspeed_config, global_args
from aigc_zoo.model_zoo.t5.llm_model import MyTransformer, PetlArguments,LoraConfig,PromptArguments


assert global_args["trainer_backend"] == "pl"

def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments, PromptArguments))
    model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(train_info_args)
    lora_args = lora_args.config
    prompt_args = prompt_args.config

    output_weight_dir = './best_ckpt'

    config_kwargs = {"torch_dtype": torch.float16}
    if global_args[ 'config_merge' ]:
        config_kwargs.update(global_args[ 'config_merge' ])
    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config(config_kwargs=config_kwargs)

    dataHelper.make_dataset_all()

    is_bf16_supported = torch.cuda.is_bf16_supported()

    precision = global_args["precision"]
    if precision == "auto":
        # 精度 根据实际情况做调整
        if is_bf16_supported:
            precision = 'bf16'
        else:
            precision = '16'

        if global_args["quantization_config"] is not None and global_args["quantization_config"].load_in_8bit:
            precision = "32"

    deepspeed_config = get_deepspeed_config(precision)
    strategy = 'ddp' if torch.cuda.device_count() > 1 else 'auto'
    if deepspeed_config is not None and len(deepspeed_config):
        strategy = DeepSpeedStrategy(config=deepspeed_config, )

    checkpoint_callback = ModelCheckpointEx(
        # monitor='loss',
        dirpath=output_weight_dir,
        save_weights_only=True,
        save_last=True,
        save_top_k=1,
        # every_n_train_steps=2000 // training_args.gradient_accumulation_steps,
        every_n_epochs=1,
        lora_args=lora_args,
        prompt_args=prompt_args,
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
        strategy=strategy,
        precision=precision  # #可以自行尝试  "32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"
        # precision='16-mixed',#混合精度训练
    )

    transformer_args = dict(config=config, model_args=model_args, training_args=training_args, lora_args=lora_args,prompt_args=prompt_args,
                             quantization_config=global_args.get('quantization_config',None),
                             device_map={"": trainer.local_rank} if trainer.world_size > 1 else "auto",
                             torch_dtype=torch.float16,
                             new_num_tokens=len(tokenizer), # 如果扩充词
                             )

    if transformer_args["quantization_config"] is None:
        transformer_args.pop("device_map")

    pl_model = MyTransformer(**transformer_args)

    config.save_pretrained(output_weight_dir)

    # 加载sft权重
    # pl_model.load_sft_weight('./best_ckpt/best.pt',is_trainable=True)

    pl_model = pl_model.float()

    def dataset_loader_filter_fn(dataset):
        print('*' * 30, 'total', len(dataset))
        return dataset


    train_datasets = dataHelper.load_distributed_random_sampler(
        dataHelper.train_files,
        with_load_memory=data_args.data_backend == 'record',
        collate_fn=dataHelper.collate_fn,
        batch_size=training_args.train_batch_size,
        drop_last=True,  # 多卡建议扔掉
        num_processes=trainer.world_size, process_index=trainer.global_rank,
        dataset_loader_filter_fn=dataset_loader_filter_fn,
        num_workers=0, #num_workers for Dataloader
    )

    if train_datasets is not None:
        trainer.fit(pl_model, train_dataloaders=train_datasets)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()


  #   if  data_args.convert_onnx:
  #       # 加载权重
  #       if not lora_args.with_lora:
  #           pl_module = MyTransformer.load_from_checkpoint('./best.pt',
  #                                                      lora_args=lora_args,
  #                                                      config=config,
  #                                                      model_args=model_args,
  #                                                      training_args=training_args)
  #           model_ = pl_module.get_llm_model()
  #           #保存权重, 可选上传至huggingface
  #           tokenizer: T5Tokenizer
  #           config: T5Config
  #           tokenizer.save_pretrained('chatyuan_finetuning')
  #           config.save_pretrained('chatyuan_finetuning')
  #           model_.save_pretrained('chatyuan_finetuning', push_to_hub = False,max_shard_size= "10GB")
  #
  #           # #转换onnx 模型
  #           # input_sample = (
  #           #     ("input_ids", torch.ones(size=(1, 128), dtype=torch.int32)),
  #           #     ("attention_mask", torch.ones(size=(1, 128), dtype=torch.int32)),
  #           #     ("decoder_input_ids", torch.ones(size=(1, 128), dtype=torch.int32)),
  #           #     ("decoder_attention_mask", torch.ones(size=(1, 128), dtype=torch.int32)),
  #           # )
  #           # input_names = ("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
  #           # output_names = ("pred_ids",)
  #           # dynamic_axes = None or {"input_ids": [0, 1], "attention_mask": [0, 1],
  #           #                         "decoder_input_ids": [0, 1], "decoder_attention_mask": [0, 1],
  #           #                         "pred_ids": [0, 1]}
  #           # pl_module.convert_to_onnx('./best.onnx',
  #           #                       input_sample=input_sample,
  #           #                       input_names=input_names,
  #           #                       output_names=output_names,
  #           #                       dynamic_axes=dynamic_axes)
  #