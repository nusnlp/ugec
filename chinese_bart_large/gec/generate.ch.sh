sudo pip3 upgrade numpy
sudo pip3 install /opt/tiger/fairseq_ch/src/src_syngec/fairseq-0.10.2
sudo pip3 install numpy==1.21.6

hadoop fs -get $HDFSHOME/dict
hadoop fs -get $HDFSHOME/UGEC/$RESULT_DIR/$model

hadoop fs -get /home/byte_ailab_litg/user/hannancao/data/UGEC/ch/BART/train-stage/$category/ungram/$HDFS_DATA_NAME
echo "Generating using fairseq-interactive: $HDFS_DATA_NAME"

fairseq-generate ${HDFS_DATA_NAME}/data-bin \
            --user-dir /opt/tiger/fairseq_ch/src/src_syngec/syngec_model/ \
            --task syntax-enhanced-translation \
            --path $model \
            --beam ${BEAM} \
            --nbest ${BEAM} \
            -s src \
            -t tgt \
            --max-tokens $MAX_TOKENS \
            --batch-size $BATCH_SIZE \
            --num-workers 12 \
            --log-format tqdm \
            --remove-bpe \
            --fp16 > $HDFS_DATA_NAME.nbest.tok

#fairseq-generate ${HDFS_DATA_NAME}/data-bin \
#                --path $model \
#                --beam ${BEAM} \
#                --nbest ${BEAM} \
#                --bpe gpt2 \
#                --source-lang src \
#                --target-lang tgt \
#                --log-format simple \
#                --remove-bpe \
#                --max-tokens $MAX_TOKENS \
#                --batch-size $BATCH_SIZE \
#                --skip-invalid-size-inputs-valid-test \
#                --fp16 --gpt2-encoder-json dict/bart/encoder.json --gpt2-vocab-bpe dict/bart/vocab.bpe > $HDFS_DATA_NAME.nbest.tok

cat $HDFS_DATA_NAME.nbest.tok | grep "^S"  | cut -f2 > $HDFS_DATA_NAME.source.tok
cat $HDFS_DATA_NAME.nbest.tok | grep "^D"  | python3 -c "import sys; x = sys.stdin.readlines(); x = [t for i, t in enumerate(x) if i % ${BEAM} == 0]; sys.stdout.writelines(x)"  | cut -f2 > $HDFS_DATA_NAME.best.sentscore
cat $HDFS_DATA_NAME.nbest.tok | grep "^D"  | python3 -c "import sys; x = sys.stdin.readlines(); x = [t for i, t in enumerate(x) if i % ${BEAM} == 0]; sys.stdout.writelines(x)"  | cut -f3 > $HDFS_DATA_NAME.best.tok


echo "Inference finished, starting GED prediction"
sudo pip3 install /opt/tiger/ged_fairseq
cd /opt/tiger/gec/chinese_roberta_ged
chmod +x ged.sh
hadoop fs -get /home/byte_ailab_litg/user/hannancao/PLM/fs_roberta.base/encoder.json
hadoop fs -get /home/byte_ailab_litg/user/hannancao/PLM/fs_roberta.base/vocab.bpe
#hadoop fs -get /home/byte_ailab_litg/user/hannancao/data/UGEC/en/BART/RoBERTa-large-BEA-train-bin
mkdir ckpt
hadoop fs -get /home/byte_ailab_litg/user/hannancao/UGEC/$HDFS_CKPT_FOLDER/checkpoint_best.pt ckpt/checkpoint_best.pt
hadoop fs -get $HDFSHOME/data/UGEC/ch/BART/gec-ged.gec.chinese.bart_large.new.crawl.ch.edit=4.both_norm.ratio=0.5.4gpu.lr=1e-5.correct.order.full/data-bin
CUDA_VISIBLE_DEVICES=0 ./ged.sh /opt/tiger/gec/$HDFS_DATA_NAME.best.tok


cd /opt/tiger/gec

hadoop fs -mkdir /home/byte_ailab_litg/user/hannancao/data/UGEC/ch/BART/train-stage/$category/ungram/$HDFS_RESULT_DIR
hadoop fs -put -f $HDFS_DATA_NAME.source.tok /home/byte_ailab_litg/user/hannancao/data/UGEC/ch/BART/train-stage/$category/ungram/$HDFS_RESULT_DIR
hadoop fs -put -f $HDFS_DATA_NAME.best* /home/byte_ailab_litg/user/hannancao/data/UGEC/ch/BART/train-stage/$category/ungram/$HDFS_RESULT_DIR


echo "Uploaded the prediction to /home/byte_ailab_litg/user/hannancao/data/UGEC/ch/BART/train-stage/$category/ungram/$HDFS_RESULT_DIR"