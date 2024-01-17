import sys
import json
import argparse
from tqdm import tqdm

from utils.tokenizer import MosesTokenzier


def convert2json(line):
    return json.dumps({'srcs': line})

# 1. detokenize
# 2. convert text to jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

# print(f'input path: {args.input}')
# print(f'output_path: {args.output}')
fin = open(args.input, 'r') if args.input else sys.stdin
fout = open(args.output, 'w') if args.output else sys.stdout

tokenizer = MosesTokenzier()

for line in tqdm(fin, desc='postprocessing'):
    sent = line.strip('\n')
    # sent = tokenizer.detokenize(sent)
    sent = convert2json(sent)
    fout.write(sent)
    fout.write('\n')

fin.close()
fout.close()
