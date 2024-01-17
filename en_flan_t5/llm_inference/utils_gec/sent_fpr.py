import sys
import argparse

from utils.tokenizer import NLTKTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--trg", type=str)
parser.add_argument("--pred", type=str)
args = parser.parse_args()

f_src = open(args.src, 'r') if args.src else sys.stdin
f_trg = open(args.trg, 'r') if args.trg else sys.stdin
f_pred = open(args.pred, 'r') if args.pred else sys.stdin

# use WordPunctTokenizer because it is fast and consistent.
tokenizer = NLTKTokenizer()

sent_T = 0
sent_FP = 0
sent_N = 0 
for src, trg, pred in zip(f_src, f_trg, f_pred):
    src = tokenizer.tokenize(src)
    trg = tokenizer.tokenize(trg)
    pred = tokenizer.tokenize(pred)
    sent_T += 1
    if src == trg:
        sent_N += 1

        if pred != trg:
            print(f"pred: {pred}")
            print(f"trg: {trg}")
            sent_FP += 1
print(f'%Negative Samples: {sent_N/sent_T:.3}')
if sent_N > 0:
    print(f'sent FPR: {sent_FP/sent_N:.3}')
else:
    print('sent FPR: NaN')

f_src.close()
f_trg.close()
f_pred.close()
