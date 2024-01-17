# interactive-generate based on DeepSpeed-Inference (批量测试别用这个脚本！！！)
# For batch generation, you should use finetune.py --do_predict to generate predictions for your test data, which is based on ZeRO-Inference.
# DeepSpeed-Inference has lower latency, while ZeRO-Inference has higher throughput.

import torch
from argparse import ArgumentParser

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import deepspeed
from deepspeed.runtime.utils import see_memory_usage

parser = ArgumentParser()

parser.add_argument("--framework", required=True,
                    type=str, choices=['seq2seq', 'clm'])
parser.add_argument("--model_name_or_path", required=True,
                    type=str, help="model_name_or_path")
parser.add_argument("--dtype", default="float16", type=str,
                    choices=["float32", "float16", "int8"], help="data-type")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
args = parser.parse_args()

local_rank = 0

data_type = getattr(torch, args.dtype)

see_memory_usage("before init", True)

model_class = AutoModelForSeq2SeqLM if args.framework=='seq2seq' else AutoModelForCausalLM
model = model_class.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
# on a small vocab and want a smaller embedding size, remove this test.
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

# TODO: deepspeed inference
# model = deepspeed.init_inference(model,
#                                  dtype=data_type,
#                                  mp_size=1,
#                                  )
pipe = pipeline(
    task='text2text-generation' if args.framework=='seq2seq' else 'text-generation',
    model=model,
    tokenizer=tokenizer,
    device=local_rank,
    max_length=128,
    num_beams=1,
    output_scores=True,
    early_stopping = True,
    do_sample=True,
    # return_full_text=False
)

see_memory_usage("after init", True)

while True:
    line = input("input: ").strip('\n')
    if line == 'exit':
        quit()
    out = pipe(line)
    print(out)

# script: 
# deepspeed --master_addr=$ARNOLD_WORKER_0_HOST --master_port=$ARNOLD_WORKER_0_PORT --num_gpus 1 interactive_generate.py --framework=seq2seq --model_name_or_path "pretrainmodel/google_flan_t5_xl"
# python3 interactive_generate.py --framework=seq2seq --model_name_or_path "pretrainmodel/google_flan_t5_xl"
# Then one sunnday, I was invited by my friend to go for a drive.
