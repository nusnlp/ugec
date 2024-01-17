import difflib
from .base import Edit
import regex as re


class DifflibDiffer():
    def __init__(self):
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d|\s| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)+"""
        )# 使用正则表达式切词，避免处理增删空格的情况。

    def get_diff(self, src_sent, trg_sent):
        
        # show diff at token-level instead of character-level for better readability
        src_tokens, src_spans = self._tokenize(src_sent)
        trg_tokens, trg_spans = self._tokenize(trg_sent)

        src_spans = src_spans + [(len(src_sent), len(src_sent)), (0, 0)] # 添加额外的token，省去处理边界情况。我真是个小天才。

        sm = difflib.SequenceMatcher(
            a=src_tokens, b=trg_tokens, autojunk=False)
        edits = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                edit = Edit(
                    ''.join(src_tokens[i1: i2]),
                    ''.join(trg_tokens[j1: j2]),
                    src_spans[i1][0],
                    src_spans[i2 - 1][1]
                )
                edits.append(edit)

        return edits

    def _tokenize(self, sent):
        tokens, spans = [], []
        for t in re.finditer(self.pat, sent):
            tokens.append(t.group())
            spans.append(t.span())
        return tokens, spans


if __name__ == "__main__":
    import sys
    import argparse
    from tqdm import tqdm
    from .base import Sentence

    parser = argparse.ArgumentParser()
    parser.add_argument("--input1", type=str)
    parser.add_argument("--input2", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    print(f'input path: {args.input1} {args.input2}')
    print(f'output_path: {args.output}')
    fin1 = open(args.input1, 'r') if args.input1 else sys.stdin
    fin2 = open(args.input2, 'r') if args.input2 else sys.stdin
    fout = open(args.output, 'w') if args.output else sys.stdout

    differ = DifflibDiffer()

    for sent1, sent2 in tqdm(zip(fin1, fin2)):
        sent1, sent2 = sent1.strip('\n'), sent2.strip('\n')
        sent = Sentence(sent1)
        edits = differ.get_diff(sent1, sent2)
        sent.set_edits(edits)
        diff_sent = sent.rebuild_text(show_diff=True)
        fout.write(diff_sent)
        fout.write('\n')
    fin1.close()
    fin2.close()
    fout.close()
