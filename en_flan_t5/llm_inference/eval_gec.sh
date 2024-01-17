FRAMEWORK='seq2seq'
DATASETS="conll14" # dataset to evaluated (bea19_test.src, conll14.src)
RESULT_DIR="pretrain_flan_t5_xxl_ckpt" # model path
ckpt=$1 # checkpoint to be evaluated
BEAM=4
SOURCE_PREFIX=" {}"
BATCH=4
# SOURCE_PREFIX="Correct grammar errors in the following sentence without changing the meaning of the original sentence: {} "
# MORE_PARA="--bf16"
# MORE_PARA=""

if [ -z "$FRAMEWORK" ]
then
    echo "FRAMEWORK is None and please choose from ['seq2seq', 'clm']"
    exit 0
fi
if [ -z "$RESULT_DIR" ]
then
    echo "RESULT_DIR is None"
    exit 0
fi
if [ -z "$DATA_EVAL" ]
then
    DATA_EVAL="data_eval"
fi
if [ -z "$BATCH" ]
then
    BATCH=128
fi
if [ -z "$BEAM" ]
then
    BEAM=1
fi
if [ -z "$SOURCE_PREFIX" ]
then
    SOURCE_PREFIX="{} "
fi
if [ -z "$REMOVE_PUNCT" ]
then
    REMOVE_PUNCT="False"
fi
if [ -z "$MORE_PARA" ]
then
    MORE_PARA=""
fi

echo "FRAMEWORK=$FRAMEWORK"
echo "RESULT_DIR=$RESULT_DIR"
echo "DATA_VER=$DATA_VER"
echo "BATCH=$BATCH"
echo "BEAM=$BEAM"
echo "SOURCE_PREFIX=$SOURCE_PREFIX"
echo "REMOVE_PUNCT=$REMOVE_PUNCT"
echo "MORE_PARA=$MORE_PARA"

# echo "Download Spacy Model..."
# sudo python3 -m spacy download en

M2SCORER=m2scorer

eval_m2scorer() {
    dataset=$1

    # detokenize and then tokenize
    input_path=$OUTPUT_DIR/${dataset}.best
    output_path=$OUTPUT_DIR/${dataset}.best.tok
    python3 utils/tokenizer.py --tokenizer=moses --mode=detokenize --input=$input_path --output=$input_path.detok1
    python3 utils/tokenizer.py --tokenizer=moses --mode=detokenize --input=$input_path.detok1 --output=$input_path.detok2
    python3 utils/tokenizer.py --tokenizer=spacy --mode=tokenize --input=$input_path.detok2 --output=$output_path

    # remove punctuation
    if [ "$REMOVE_PUNCT" = "True" ]
    then
        python3 -m utils_gec.remove_punct --input1=$DATA_EVAL_DIR/${dataset}.src --input2=$OUTPUT_DIR/${dataset}.best.tok --output=$OUTPUT_DIR/${dataset}.best.punct
    fi
    echo "REMOVE_PUNCT: $REMOVE_PUNCT"
}



gen() {
    dataset=$1
    OUTPUT_DIR=$3

    echo "dataset: $dataset"

    input=$OUTPUT_DIR/${dataset}.src

    # detokenize and convert text to json
    python3 postprocess.py --input=${dataset}.src --output=$input

    test_args=(
        # [generate]
        --do_predict
        --predict_with_generate
        --per_device_eval_batch_size $BATCH
        --num_beams $BEAM
        # [output]
        --test_file $input
        --predict_output_file $OUTPUT_DIR/${dataset}.best
        --output_dir $OUTPUT_DIR
        # [deepspeed]
        --deepspeed ds_config_zero3.json
        # [model]
        --framework $FRAMEWORK
        --model_name_or_path $2
        --source_lang srcs --target_lang trgs
        --max_source_length 512 --max_target_length 512
        --source_prefix "$SOURCE_PREFIX"
        # [other]
        ${MORE_PARA}
    )

    python3 -u -m torch.distributed.run \
        --nproc_per_node 4 \
        --nnodes 1 \
        --rdzv_backend c10d \
        --rdzv_conf read_timeout=3600 \
        finetune.py "${test_args[@]}"


}

standard_eval_datasets() {

  if [ -z "$DATASETS" ]
  then
      echo "DATASETS is None. Please fill in DATASETS with datasets seperated by comma"
      exit 0
  fi
  model=$1
  RESULT_DIR=$model
  OUTPUT_DIR=output_${model}
  mkdir -p $OUTPUT_DIR
  # python3 utils/shard_model.py --input $model --framework seq2seq --output $OUTPUT_DIR/$model-shard
  echo "Generating result of $model to $OUTPUT_DIR"

  for dataset in ${DATASETS//,/ }
  do
    gen $dataset $OUTPUT_DIR/$model-shard $OUTPUT_DIR
  done
}

standard_eval_datasets $ckpt