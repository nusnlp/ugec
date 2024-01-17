# FRAMEWORK='clm'
# DATASETS="conll14"
# RESULT_DIR="llama_65b_hf.clang8"
# BATCH=1
# BEAM=4
# SOURCE_PREFIX = " {}"
# # SOURCE_PREFIX="Correct grammar errors in the following sentence without changing the meaning of the original sentence: {} "
# MORE_PARA="--bf16"

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

echo "Download Spacy Model..."
SPACY_MODEL=en_core_web_md-3.4.0.tar.gz
hadoop fs -get $HDFSHOME/tools/$SPACY_MODEL
sudo pip3 install $SPACY_MODEL

DATA_EVAL=en
echo "Download Evaluation Data..."
hadoop fs -get $HDFSHOME/data/test_data/$DATA_EVAL
DATA_EVAL_DIR=$DATA_EVAL

M2SCORER=m2scorer

eval_m2scorer() {
    dataset=$1

    # detokenize and then tokenize
    input_path=$OUTPUT_DIR/${dataset}.best
    output_path=$OUTPUT_DIR/${dataset}.best.tok
    python3 utils/tokenizer.py --tokenizer=moses --mode=detokenize --input=$input_path --output=$input_path.detok1
    python3 utils/tokenizer.py --tokenizer=moses --mode=detokenize --input=$input_path.detok1 --output=$input_path.detok2
    python3 utils/tokenizer.py --tokenizer=spacy --mode=tokenize --input=$input_path.detok2 --output=$output_path
    rm $input_path.detok1
    rm $input_path.detok2

    # remove punctuation
    if [ "$REMOVE_PUNCT" = "True" ]
    then
        python3 -m utils_gec.remove_punct --input1=$DATA_EVAL_DIR/${dataset}.src --input2=$OUTPUT_DIR/${dataset}.best.tok --output=$OUTPUT_DIR/${dataset}.best.punct
        mv $OUTPUT_DIR/${dataset}.best.punct $OUTPUT_DIR/${dataset}.best.tok
    fi

    # eval via m2scorer
    python2 $M2SCORER/m2scorer.py -v $OUTPUT_DIR/${dataset}.best.tok $DATA_EVAL_DIR/${dataset}.m2 > $OUTPUT_DIR/${dataset}.m2score.log
    tail -n 10 $OUTPUT_DIR/${dataset}.m2score.log
    python3 -m utils_gec.sent_fpr --src=$DATA_EVAL_DIR/${dataset}.src --trg=$DATA_EVAL_DIR/${dataset}.trg --pred=$OUTPUT_DIR/${dataset}.best.tok >> $OUTPUT_DIR/${dataset}.m2score.log
    tail -n 2 $OUTPUT_DIR/${dataset}.m2score.log
}



gen() {
    dataset=$1
    OUTPUT_DIR=$3

    echo "dataset: $dataset"

    input=$OUTPUT_DIR/${dataset}.src

    # detokenize and convert text to json
    python3 postprocess.py --input=$DATA_EVAL_DIR/${dataset}.src --output=$input

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
        --nproc_per_node $ARNOLD_WORKER_GPU \
        --nnodes $ARNOLD_WORKER_NUM \
        --rdzv_endpoint $ARNOLD_WORKER_0_HOST:2222 \
        --rdzv_backend c10d \
        --rdzv_conf read_timeout=3600 \
        finetune.py "${test_args[@]}"


    eval_m2scorer ${dataset}

}

standard_eval_datasets() {

  if [ -z "$DATASETS" ]
  then
      echo "DATASETS is None. Please fill in DATASETS with datasets seperated by comma"
      exit 0
  fi

  echo "Download Checkpoints..."
  model=checkpoint-$1
  mkdir -p $RESULT_DIR
  if [ -d "${RESULT_DIR}/${model}" ]; then
    echo "Detect ${RESULT_DIR}/${model}"
  else
    echo "Checkpoint ${RESULT_DIR}/${model} doesn't exist. Try to download from remote: $HDFSHOME/LLM/$RESULT_DIR/$model."
    hadoop fs -get $HDFSHOME/LLM/$RESULT_DIR/$model $RESULT_DIR/
  fi

  OUTPUT_DIR=$RESULT_DIR/output_${model}
  mkdir -p $OUTPUT_DIR

  echo "Generating result of $model to $OUTPUT_DIR"

  for dataset in ${DATASETS//,/ }
  do
    gen $dataset $RESULT_DIR/$model $OUTPUT_DIR
  done

  echo "Uploading outputs to ${HDFSHOME}/LLM/${RESULT_DIR}/${OUTPUT_DIR}"
  hadoop fs -mkdir $HDFSHOME/LLM/$RESULT_DIR
  hadoop fs -put -f $OUTPUT_DIR $HDFSHOME/LLM/$RESULT_DIR/
  rm -rf $RESULT_DIR/$model
}