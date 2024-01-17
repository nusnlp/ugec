import re
import sys
import argparse
from tqdm import tqdm

detok_pat = [
    (" n't ", "n't "),
    (" n' t ", "n't "),
    (" 'm ", "'m "),
    (" ' m ", "'m "),
    (" 'll ", "'ll "),
    (" ' ll ", "'ll "),
    (" 're ", "'re "),
    (" ' re ", "'re "),
    (" 's ", "'s "),
    (" ' s ", "'s "),
    (" 't ", "'t "),
    (" ' t ", "'t "),
    ("' ve ", "'ve "),
    ("' d ", "'d "),
    (" i. e. ", " i.e. "),
    ("' '", " '' "),
    (r"\s*-\s*", "-"),
    (r"\s*/\s*", "/"),
    (r"([0-9])\s*:\s*([0-9])", r"\1:\2"),
    (r"([0-9])\s*,\s*([0-9])", r"\1,\2"),
    (r"([0-9])\s*\.\s*([0-9])", r"\1.\2"),
]


class MosesTokenzier:
    def __init__(self) -> None:
        from mosestokenizer import MosesTokenizer, MosesDetokenizer
        self.tokenizer = MosesTokenizer('en')
        self.detokenizer = MosesDetokenizer('en')

    def tokenize(self, sent):
        tok_sent = self.tokenizer(sent)
        return ' '.join(tok_sent)

    def detokenize(self, sent):
        detok_sent = self.detokenizer(sent.split())
        for p in detok_pat:
            detok_sent = re.sub(p[0], p[1], detok_sent)
        return detok_sent


class NLTKTokenizer:
    def __init__(self) -> None:
        from nltk.tokenize import TreebankWordDetokenizer
        from nltk.tokenize import WordPunctTokenizer
        self.tokenizer = WordPunctTokenizer()
        self.detokenizer = TreebankWordDetokenizer()

    def tokenize(self, sent):
        tokens = self.tokenizer.tokenize(sent)
        tok_sent = ' '.join(tokens)
        return tok_sent

    def detokenize(self, sent):
        detok_sent = self.detokenizer.tokenize(sent.split())
        for p in detok_pat:
            detok_sent = re.sub(p[0], p[1], detok_sent)
        return detok_sent


class SpacyTokenizer:
    def __init__(self):
        import spacy
        self.nlp = spacy.load('en_core_web_md')

    def tokenize(self, sent):
        sent = self.nlp(sent, disable=['tagger', 'parser', 'ner'])
        tokens = [t.text for t in sent]
        tok_sent = ' '.join(tokens)
        return tok_sent

    def detokenize(self, sent):
        raise NotImplementedError


class TransformerTokenizer:
    def __init__(self, model_name) -> None:
        from transformers import AutoTokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            do_lower_case=False
        )

    def tokenize(self, sent):
        tokens = self.tokenizer.tokenize(sent)
        tok_sent = ' '.join(tokens)
        return tok_sent

    def detokenize(self, sent):
        detok_sent = self.tokenizer.convert_tokens_to_string(sent.split())
        for p in detok_pat:
            detok_sent = re.sub(p[0], p[1], detok_sent)
        return detok_sent

class RegexTokenizer:
    def __init__(self) -> None:
        import regex as re
        self.re = re
        self.pat = self.re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)+|\s"""
        )
    def tokenize(self, sent):
        tokens = [t.strip() for t in self.re.findall(self.pat, sent)]
        return ' '.join(tokens)
    
    def detokenize(self, sent):
        raise NotImplementedError

class CharacterTokenizer:
    def __init__(self) -> None:
        from fairseq.data.encoders.characters import Characters
        self.encoder = Characters(None)
    def tokenize(self, sent):
        return self.encoder.encode(sent)
    def detokenize(self, sent):
        return self.encoder.decode(sent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default='spacy')
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--mode", type=str, default='tokenize')
    parser.add_argument("--model_name", type=str, default='bert-base-cased')
    args = parser.parse_args()

    print(f'input path: {args.input}')
    print(f'output_path: {args.output}')
    fin = open(args.input, 'r') if args.input else sys.stdin
    fout = open(args.output, 'w') if args.output else sys.stdout

    if args.tokenizer == 'nltk':
        tokenizer = NLTKTokenizer()
    elif args.tokenizer == 'transformers':
        tokenizer = TransformerTokenizer(args.model_name)
    elif args.tokenizer == 'moses':
        tokenizer = MosesTokenzier()
    elif args.tokenizer == 'regex':
        tokenizer = RegexTokenizer()
    elif args.tokenizer == 'character':
        tokenizer = CharacterTokenizer()
    else:
        tokenizer = SpacyTokenizer()

    for line in tqdm(fin, desc=f'[{args.mode}]'):
        sent = line.strip('\n')
        if args.mode == 'tokenize':
            sent = tokenizer.tokenize(sent)
        else:
            sent = tokenizer.detokenize(sent)
        fout.write(sent)
        fout.write('\n')

    fin.close()
    fout.close()
