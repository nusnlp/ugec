import json
import argparse
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--framework", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

MODEL_CLASS = AutoModelForSeq2SeqLM if args.framework == 'seq2seq' else AutoModelForCausalLM
model = MODEL_CLASS.from_pretrained(args.input)
tokenizer = AutoTokenizer.from_pretrained(args.input)
model.save_pretrained(args.output, max_shard_size="10GB")
tokenizer.save_pretrained(args.output)