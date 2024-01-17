from transformers import pipeline
import sys
import os
import torch

import re
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()

def handle_dounble_quote(sent):
    cur_str = ''
    exp_left = True
    ignore_space = False
    for char in sent:
        if char == '"':
            if exp_left: #this is a left "
                cur_str = cur_str.rstrip() + ' "'
                exp_left = (not exp_left)
                ignore_space = True
            else: #this is a right "
                cur_str = cur_str.rstrip() + '" '
                exp_left = (not exp_left)
                ignore_space = False
        else:
            if ignore_space: #expecting right
                if char == ' ':
                    continue
                else:
                    cur_str = cur_str + char
                    ignore_space = False
            else:
                cur_str = cur_str + char
    cur_str = cur_str.strip()
    cur_str = re.sub(r'[ ]+', ' ', cur_str)
    return cur_str

def postprocess_space(sent):
    sent = re.sub(r'[ ]+\.', '.', sent)
    sent = re.sub(r'[ ]+,', ',', sent)
    sent = re.sub(r'[ ]+!', '!', sent)
    sent = re.sub(r'[ ]+\?', '?', sent)
    sent = re.sub(r'\([ ]+', '(', sent)
    sent = re.sub(r'[ ]+\)', ')', sent)
    sent = re.sub(r' \'s( |\.|,|!|\?)', r"'s\1", sent)
    sent = re.sub(r'n \'t( |\.|,|!|\?)', r"n't\1", sent)
    return sent

def detokenize_sent(sent):
    #Clean raw sent
    sent = re.sub(r'\' s ', '\'s ', sent)
    toks = sent.split()
    if len([1 for t in toks if t=="'"]) % 2 == 0:
        toks = ['"' if t=="'" else t for t in toks]
    sent = ' '.join(toks)
    #
    sents = sent_tokenize(sent)
    final_sents = []
    for _sent in sents:
        _sent = detokenizer.detokenize(_sent.split())
        res = handle_dounble_quote(_sent)
        if res == -1:
            print ('unbalanced double quote')
            print (_sent)
        else:
            _sent = res
        final_sents.append(_sent)
    sent = ' '.join(final_sents)
    sent = postprocess_space(sent)
    return sent

import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS, HYPHENS
from spacy.util import compile_infix_regex
from spacy.lang.en import English
nlp = English()

def get_tokenizer_gec(nlp):
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )
    infix_re = compile_infix_regex(infixes)
    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)


def get_tokenizer_bea19(nlp):
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )
    infix_re = compile_infix_regex(infixes)
    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)


tokenizer_gec = get_tokenizer_gec(nlp)
tokenizer_bea19 = get_tokenizer_bea19(nlp)


def spacy_tokenize_gec(text):
    nlp.tokenizer = tokenizer_gec
    return [str(w) for w in nlp(text)]

def spacy_tokenize_bea19(text):
    nlp.tokenizer = tokenizer_bea19
    return [str(w) for w in nlp(text)]

model_name = sys.argv[1]
model_path = os.path.join("/opt/tiger/LLMFinetune", model_name)
output_name = sys.argv[2]
prompt_idx = int(sys.argv[3])
prompt_list = ["Correct this to standard English: ", \
        "Correct grammar errors in the following sentence: ", \
        "Correct grammar errors in the following sentence without changing the meaning of the original sentence: ", \
        "Make minimal changes to make the following sentence grammatical, only make a change if it is necessary: "]
if prompt_idx < 4:
    prompt = prompt_list[prompt_idx]

p=pipeline("text-generation", model_path, torch_dtype=torch.float16, device_map="auto", max_length=1024)
input_name=sys.argv[4]
test_src = open(os.path.join("/opt/tiger/LLMFinetune/en/conll14_split", input_name)).readlines()
# test_src = open("/opt/tiger/LLMFinetune/en/test_conll14.src").readlines()

# print(len(test_src))
# print(prompt)
# print("==============")
with open(output_name, "w+", encoding="utf-8") as out_f:
    for idx in range(len(test_src)):
        if prompt_idx == 5:
            input_text = "Please identify and correct any grammatical errors in the following sentence indicated by <input> " + detokenize_sent(test_src[idx].replace("\n", "")) + " </input> tag, you need to comprehend the sentence as a whole before identifying and correcting any errors step by step while keeping the original sentence structure unchanged as much as possible. Afterward, output the corrected version directly without any explanations. Remember to format your corrected output results with the tag <output> Your Corrected Version </output>. Please start: <input> " + detokenize_sent(test_src[idx].replace("\n", "")) + " </input>: "
        elif prompt_idx == 4:
            input_text = "Reply with a corrected version of the input sentence with all grammatical and spelling errors fixed. If there are no errors, reply with a copy of the original sentence. Input sentence : " + detokenize_sent(test_src[idx].replace("\n", "")) +  " Corrected sentence: "
        else:
            input_text = prompt + detokenize_sent(test_src[idx].replace("\n", ""))
        output = p(input_text)[0]['generated_text'].replace(input_text, "")
        # output = p(input_text)[0]['generated_text']
        toknized_output_all = " ".join(spacy_tokenize_gec(output))
        toknized_output_list = toknized_output_all.split("\n")
        # print(toknized_output_list)
        # break
        if toknized_output_list[0] == "" or toknized_output_list[0] == ". ":
            if len(toknized_output_list) > 1:
                toknized_output = toknized_output_list[1]
            else:
                toknized_output = toknized_output_list[0]
        else:
            toknized_output = toknized_output_list[0]
        out_f.write(input_text + "\t" + str(idx) + "\t" + toknized_output + "\n")