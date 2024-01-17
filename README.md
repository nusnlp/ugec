# Unsupervised Grammatical Error Correction Rivaling Supervised Methods

> Hannan Cao, Liping Yuan, Yuchen Zhang, Hwee Tou Ng. Unsupervised Grammatical Error Correction Rivaling Supervised Methods. In EMNLP 2023. 

The program is tested under pytorch 1.7.1, CUDA version 11.7 

## Training Data & Checkpoints
[GEC training data](https://drive.google.com/drive/folders/1c1xNjD7ORGaY9P3vuy1w-q_f60WkGC_D?usp=sharing);
[GEC model checkpoints](https://drive.google.com/drive/folders/1TZNbuEwjifTVqKldfpkXl264CLkHNlXw?usp=sharing);
## English GEC

### Flan-T5-xxl
1. Please store all the downloaded checkpoint and data for Flan-T5-xxl in this folder: en_flan_t5/llm_finetune
2. Install the requirement.txt inside en_flan_t5 folder
Train:
```
bash train.sh
```
Inference: go to en_flan_t5/llm_inference folder
```
bash eval_gec.sh your/ckpt/name
```
### BART-base
1. Please store all the downloaded checkpoint and data for BART-base in this folder: en_fairseq_train
2. Install the requirement.txt inside en_fairseq_train folder
Train:
```
cd gec
bash train.sh path/to/the/model/to/be/restored path/to/data-bin/folder output_path
```
Inference:
```
bash new_generate.sh path/to/model/ckpt testing/input/path
```
## Chinese GEC
1. Please store all the downloaded checkpoint and data for BART-base in this folder: chinese_bart_large
2. Install the requirement.txt inside chinese_bart_large folder
Train:
```
cd gec
bash train_ch.sh
```
Inference:
```
cd gec
bash test_ch.sh
```
## Citation

If you found our paper or code useful, please cite as:

```
@inproceedings{cao-etal-2023-unsupervised,
    title = "Unsupervised Grammatical Error Correction Rivaling Supervised Methods",
    author = "Cao, Hannan  and
      Yuan, Liping  and
      Zhang, Yuchen  and
      Ng, Hwee Tou",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.185",
    doi = "10.18653/v1/2023.emnlp-main.185",
    pages = "3072--3088",
}
```

If you encounter any problem with the code, please contact caoh@u.nus.edu .