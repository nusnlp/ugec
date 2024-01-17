# LLM finetune based on DeepSpeed and Transformers

环境配置请参考 Arnold Task: https://arnold.byted.org/task/3689357

**环境配置注意事项**

torch 版本 >= 1.12，否则 wandb 会报错 ["histogram_cpu" not implemented for 'BFloat16' when using deepspeed and reporting to wandb](https://github.com/huggingface/transformers/issues/16689)

GPU 数量的选择：GPU 数量过少会导致模型占用显存比例过大，GPU 利用率很低；GPU 数量过多会导致 GPU 的通信成本太高。10B 以内的模型尽量不要多机（除非排不到A100）。

模型 GPU 显存占用计算参考：[LLM Training](https://bytedance.feishu.cn/docx/OvN8dUfqeoGWoyxezKVcPwoQndb)

llama 的 transformers 版本：`pip3 install git+https://github.com/huggingface/transformers.git@refs/pull/21955/merge` 或者使用 [SCM](https://cloud.bytedance.net/scm/detail/280433/versions) 版本。


## 文件说明

`train.sh` 训练脚本。

`eval_gec.sh` GEC 任务的测试脚本。包括生成模型预测结果及使用 `m2scorer` 评估。

`interactive_generate.py` interactive generation。批量测试别用这个脚本！批量测试请使用 `finetune.py --do_predict`（参考 `eval_gec.sh`）

## 一些使用说明

merlin-wandb 上会有相关的训练指标的记录，可以去看。

[llm-checkpoints](https://arnold.byted.org/volume/5476) 上下载好了一些 LLM checkpoints。

HDFS 每次上传/下载 checkpoints 都太慢了，建议申请自己的 bytenas，checkpoints 写到 bytenas 上。

## Supported LLMs

理论上支持所有 transformers 上的模型。

- **FLAN-T5**

    **注意事项**

    1. FLAN-T5 使用 fp16 会报[溢出错误](https://github.com/microsoft/DeepSpeed/issues/2169#issuecomment-1203472651)。
    因此，V100 的机器只能设置 fp32，A100 的机器可以设置 fp32/bf16。

    2. DeepSpeed 中，`global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus`

    **训练设置**
    ```bash
    bash train.sh
    # google/flan-t5-xl 
    # A100-SXM-80GB * 1
    # 参数量 3B，模型+优化器共需显存约 48G
    MODEL_NAME="google_flan_t5_xl"
    FRAMEWORK='seq2seq'
    DATA_VER="spell_lang8"
    RESULT_DIR="google_flan_t5_xl-spell_lang8"
    BATCH=16
    GRADIENT_ACCUMULATION_STEPS=4
    TRAIN_STEPS=1000
    SAVE_STEPS=100
    MORE_PARA="--bf16"
    ```

    **推理设置**
    ```bash
    source eval_gec.sh
    standard_eval_datasets 500
    # google/flan-t5-xl 
    # A100-SXM-80GB * 1
    FRAMEWORK='seq2seq'
    DATASETS="spell_formal"
    RESULT_DIR="google_flan_t5_xl-spell_lang8"
    BATCH=128
    BEAM=4
    MORE_PARA="--bf16"
    ```
   
- **GPT2**
    
    **注意事项**

    1. GPT2 使用 absolute position encoding。设置 `tokenizer.padding_side = "left"`。transformers creates postion_ids on the fly for batch generation for gpt2 [参考这个PR](https://github.com/huggingface/transformers/pull/7552#event-3876130796)。
    

- **BLOOM**

    **训练设置**
    ```bash
    bash train.sh
    # bigscience/bloom-3b 
    # A100-SXM-80GB * 1
    # 参数量 3B，模型+优化器共需显存约 48G
    MODEL_NAME="bigscience_bloom_3b"
    FRAMEWORK='clm'
    DATA_VER="spell_lang8"
    RESULT_DIR="bigscience_bloom_3b-spell_lang8"
    BATCH=16
    GRADIENT_ACCUMULATION_STEPS=4
    TRAIN_STEPS=1000
    SAVE_STEPS=500
    MORE_PARA="--bf16"
    ```

## FixBug

- [ ] 模型内存爆炸问题

    问题描述：8卡机器训练时，8个进程同时 load checkpoint，内存占用量是一个 checkpoint 的 8 倍，会导致内存不够。解决办法：先多申请点内存；doing：save/load sharded checkpoints。

## TODO
- [ ] support Chinese LLMs
- [ ] support Zero-Offload
- [ ] support prompt tuning
- [ ] support RLHF tuning