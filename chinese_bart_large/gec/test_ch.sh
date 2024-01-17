if [ -z "$BPE" ]
then
    BPE=none

fi

if [ -z "$BPE_CODE" ]
then
    BPE_CODE=dict/trans/bpe_code.trg.dict_bpe8000
fi



if [ -z "$cpu_num" ]
then
    cpu_num=4

fi


if [ -z "$FAIRSEQ" ]
then
    FAIRSEQ=/opt/tiger/fairseq/src/src_syngec/fairseq-0.10.2
fi


if [ -z "$SKIP_GEN" ]
then
    SKIP_GEN=False
fi


if [ -z "$EVAL_BEST_CHECKPOINT" ]
then
    EVAL_BEST_CHECKPOINT=True
fi



if [ -z "$EPOCH" ]
then
    EPOCH=20
fi


if [ -z "$seed" ]
then
    seed=2222
fi


if [ -z "$BEAM" ]
then
    BEAM=4
fi

if [ "$BPE" = "subword_nmt" ]; then
    TEST_MORE_PARA=$TEST_MORE_PARA" --bpe-codes $BPE_CODE"
fi


if [ -z "$DATA_EVAL" ]
then
    DATA_EVAL=data_eval
fi


DATA_EVAL_DIR=$DATA_EVAL


M2SCORER=m2scorer



gen() {
    EVAL=$1
    dataset=$2
    OUTPUT_DIR=$3
    revert_split=$4
    model=$5
    dataset=nlpcc
    DATA_EVAL_DIR=ch

    if [ -z "$model" ]
    then
          model=checkpoint_best.pt
    fi

    echo "dataset: $dataset"

    input=$DATA_EVAL_DIR/${dataset}.src.char
    cp $input $OUTPUT_DIR/${dataset}.src.char


    echo "Generating using fairseq-interactive: $input, ${RESULT_DIR}/$model , ${PROCESSED_DIR}"


    if [ "$SKIP_GEN" = "False" ]
    then

          fairseq-interactive $PROCESSED_DIR \
            --user-dir /home/project/11003628/hannan/ch_fairseq/chinese_bart_large/fairseq/src/src_syngec/syngec_model/ \
            --task syntax-enhanced-translation \
            --path $model \
            --beam ${BEAM} \
            --nbest ${BEAM} \
            -s src \
            -t tgt \
            --buffer-size 10000 \
            --batch-size 64 \
            --num-workers 12 \
            --log-format tqdm \
            --remove-bpe \
            --fp16 \
            --output_file $OUTPUT_DIR/${dataset}.nbest.char.tok < $OUTPUT_DIR/${dataset}.src.char

    else
        echo "Skip Generation due to SKIP_GEN=$SKIP_GEN"
    fi
    cat $OUTPUT_DIR/${dataset}.nbest.char.tok | grep "^D-" | python3 -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${BEAM} == 0) ]); print(x)" | cut -f3 > $OUTPUT_DIR/${dataset}.best.char.tok
    sed -i '$d' $OUTPUT_DIR/${dataset}.best.char.tok
    cat $OUTPUT_DIR/${dataset}.nbest.char.tok | grep "^D-" | python3 -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${BEAM} == 0) ]); print(x)" | cut -f2 > $OUTPUT_DIR/${dataset}.best.char.sentscore

    python3 postprocess_ch.py $OUTPUT_DIR/${dataset}.src.char $OUTPUT_DIR/${dataset}.best.char.tok $OUTPUT_DIR/${dataset}.best.char.tok.post_process
    python3 segment_pkunlp.py $OUTPUT_DIR/${dataset}.best.char.tok.post_process.segment.output < $OUTPUT_DIR/${dataset}.best.char.tok.post_process

    python2 $M2SCORER/m2scorer.py -v $OUTPUT_DIR/${dataset}.best.char.tok.post_process.segment.output $DATA_EVAL_DIR/nlpcc.gold.01 > $OUTPUT_DIR/${dataset}.m2score.log
    tail -n 3 $OUTPUT_DIR/${dataset}.m2score.log
}

for ck in $(seq 1 20)
do
    BEAM=4
    PROCESSED_DIR=data-bin
    model=checkpoint$ck.pt
    OUTPUT_DIR=output_${model}
    mkdir -p $OUTPUT_DIR
    echo "Generating result on model $model to $OUTPUT_DIR"
    gen EVAL_m2 valid $OUTPUT_DIR True $model
done

BEAM=4
PROCESSED_DIR=data-bin
model=checkpoint_best.pt
OUTPUT_DIR=output_${model}
mkdir -p $OUTPUT_DIR
echo "Generating result on model $model to $OUTPUT_DIR"
gen EVAL_m2 valid $OUTPUT_DIR True $model
