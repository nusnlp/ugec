import argparse
import sys
from string import punctuation, whitespace
from tqdm import tqdm
from .differ import DifflibDiffer
from .base import Sentence

def is_punct_edit(edit):
    whitelist = punctuation + whitespace
    orig_text = edit.orig_text
    correct_text = edit.correct_text
    for c in whitelist:
        orig_text = orig_text.replace(c, '')
    for c in whitelist:
        correct_text = correct_text.replace(c, '')
    if orig_text == correct_text or orig_text == "":
        return True
    return False


def process_line(sent1, sent2):
    sent1, sent2 = sent1.strip(), sent2.strip()
    
    sent = Sentence(sent1)
    edits = differ.get_diff(sent1, sent2)
    edits = [e for e in edits if not is_punct_edit(e)]

    sent.set_edits(edits)
    out = sent.rebuild_text()
    return out

differ = DifflibDiffer()

parser = argparse.ArgumentParser()
parser.add_argument("--input1", type=str)
parser.add_argument("--input2", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

print(f'input paths: {args.input1}, {args.input2}')
print(f'output_path: {args.output}')
fin1 = open(args.input1, 'r') if args.input1 else sys.stdin
fin2 = open(args.input2, 'r') if args.input2 else sys.stdin
fout = open(args.output, 'w') if args.output else sys.stdout


for sent1, sent2 in tqdm(zip(fin1, fin2), desc='[remove punctuation]'):
    out = process_line(sent1, sent2)
    fout.write(out)
    fout.write('\n')

fin1.close()
fin2.close()
fout.close()
