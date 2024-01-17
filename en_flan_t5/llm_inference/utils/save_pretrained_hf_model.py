from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

name = 'decapoda-research/llama-65b-hf'
tokenizer = AutoTokenizer.from_pretrained(name)

flag = True
while flag:
    try:
        model = AutoModelForCausalLM.from_pretrained(name)
        flag = False
    except:
        print('retrying...')

tokenizer.save_pretrained('/mnt/bn/llm-checkpoints/decapoda_research_llama_65b_hf')
model.save_pretrained('/mnt/bn/llm-checkpoints/decapoda_research_llama_65b_hf')