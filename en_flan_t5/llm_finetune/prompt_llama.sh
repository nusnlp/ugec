export https_proxy=http://10.20.47.147:3128
pip3 install /opt/tiger/llama_transformers
#model="decapoda_research_llama_7b_hf"
#split="split.0"
#prompt="1"
mkdir /home/tiger/nltk_data
mkdir /home/tiger/nltk_data/tokenizers
hadoop fs -get $HDFSHOME/tools/punkt /home/tiger/nltk_data/tokenizers
hadoop fs -get $HDFSHOME/data/test_data/en
hadoop fs -get $HDFSHOME/pretrainmodel/$model
SPACY_MODEL=en_core_web_md-3.4.0.tar.gz
hadoop fs -get $HDFSHOME/tools/$SPACY_MODEL
sudo pip3 install $SPACY_MODEL


output_name="prompt$prompt.$model.$split"
python3 prompt_llama.py $model $output_name $prompt $split

hadoop fs -mkdir "$HDFSHOME/prompt_output/prompt$prompt.$model"
hadoop fs -put -f $output_name "$HDFSHOME/prompt_output/prompt$prompt.$model"
echo "hadoop fs -put -f $output_name $HDFSHOME/prompt_output/prompt$prompt.$model finished"