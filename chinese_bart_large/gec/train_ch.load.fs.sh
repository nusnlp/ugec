
sudo pip3 uninstall -y numpy

if [ -z "$FAIRSEQ" ]
then
    FAIRSEQ=/opt/tiger/fairseq/src/src_syngec/fairseq-0.10.2
fi

if [ -z "$EPOCH" ]
then
    EPOCH=20
fi


if [ -z "$seed" ]
then
    seed=2222
fi

# just set RESTORE_MODEL=none if do not use pretrain model
if [ -z "$RESTORE_MODEL" ]
then
    RESTORE_MODEL=pretrainmodel/${MODEL_ARCH}_${Lang}/model.pt
fi


if [ -z "$DO_TRAIN" ]
then
    DO_TRAIN=True
fi


if [ -z "$BATCH" ]
then
    BATCH=1024
fi


if [ -z "$SAVE_INTERVAL" ]
then
    SAVE_INTERVAL=1
fi

if [ -z "$DROP_OUT" ]
then
    DROP_OUT=0.3
fi

if [ -z "$CLIP_NORM" ]
then
    CLIP_NORM=0.1
fi

if [ -z "$LR" ]
then
    LR=1e-5
fi

if [ -z "$UPDATE_FRE" ]
then
    UPDATE_FRE=1e-5
fi

sudo pip3 install $FAIRSEQ

export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_SOCKET_IFNAME=eth0,eth2
export NCCL_IB_DISABLE=0
export NCCL_BLOCKING_WAIT=1

# set -e
# set -x

WORKER_ID=$ARNOLD_ID

IFS=','
HOST_CONFIGS=($ARNOLD_WORKER_HOSTS)
WORKER_COUNT=(${#HOST_CONFIGS[@]})
unset IFS

IFS=':'
WORKER0=(${HOST_CONFIGS[0]})
WORKER0_IP=${WORKER0[0]}
unset IFS

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

MASTER_ADDR=$WORKER0_IP

if [ -z "$MASTER_PORT" ]
then
  MASTER_PORT=9099
fi


echo "DATA_VER=$DATA_VER"
echo "MODEL_ARCH=$MODEL_ARCH"
echo "Lang=$Lang"
echo "BPE=$BPE"
echo "cpu_num=$cpu_num"
echo "FAIRSEQ=$FAIRSEQ"
echo "EPOCH=$EPOCH"
echo "seed=$seed"
echo "RESULT_DIR=$RESULT_DIR"
echo "DO_TRAIN=$DO_TRAIN"
echo "BATCH=$BATCH"
echo "WORKER_COUNT=$WORKER_COUNT"
echo "GPU_COUNT=$GPU_COUNT"
echo "WORKER_ID=$WORKER_ID"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "DROP_OUT=$DROP_OUT"
echo "CLIP_NORM=$CLIP_NORM"
echo "LR=$LR"
echo "UPDATE_FRE=$UPDATE_FRE"
echo "LR_SHCE=$LR_SHCE"

#mkdir -p PLM
hadoop fs -get $HDFSHOME/${RESTORE_MODEL}
#  hadoop fs -get $HDFSHOME/PLM/gpt2 PLM/gpt2


# PROCESSED_DIR=process_${DATA_VER}_${MODEL_ARCH}_${Lang}
# mkdir -p $PROCESSED_DIR
hadoop fs -get $HDFSHOME/data/UGEC/ch/BART/$HDFS_PROCESSED_DIR
#hadoop fs -get $HDFSHOME/data/UGEC/en/BART/LM-Critic-Pretrain-GED-bin


echo "PROCESSED_DIR:$PROCESSED_DIR"
echo "RESTORE_MODEL:$RESTORE_MODEL"
echo "MODEL_DIR:$MODEL_DIR"
echo "MORE_PARA:$MORE_PARA"


train() {
  fairseq-train $PROCESSED_DIR \
    --save-dir $MODEL_DIR \
    --user-dir /opt/tiger/fairseq/src/src_syngec/syngec_model/ \
    --restore-file $RESTORE_LOCAL_MODEL \
    --task syntax-enhanced-translation \
    --arch syntax_enhanced_bart_large \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens $BATCH \
    --optimizer adam \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --lr $LR \
    --warmup-updates $WARMUP_UPDATES \
    -s src \
    -t tgt \
    --lr-scheduler polynomial_decay \
    --clip-norm 1.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch $EPOCH \
    --share-all-embeddings \
    --adam-betas '(0.9,0.999)' \
    --log-format tqdm \
    --find-unused-parameters \
    --fp16 \
    --seed $seed $MORE_PARA

   hadoop fs -mkdir $HDFSHOME/UGEC/$HDFS_MODEL_DIR
   hadoop fs -put -f $MODEL_DIR/checkpoint*.pt $HDFSHOME/UGEC/$HDFS_MODEL_DIR/
   echo "Upload checkpoint_best.pt to $HDFSHOME/UGEC/$HDFS_MODEL_DIR"

   #  fairseq-train $PROCESSED_DIR \
#     --save-dir $MODEL_DIR \
#     --restore-file $RESTORE_LOCAL_MODEL \
#     --task $TASK \
#     --layernorm-embedding --share-all-embeddings \
#     --share-decoder-input-output-embed \
#     --required-batch-size-multiple 1 \
#     --arch $MODEL_ARCH \
#     --criterion $CRITERION \
#     --max-epoch $EPOCH \
#     --seed $seed \
#     --log-format simple \
#     --max-tokens $BATCH \
#     --dropout $DROP_OUT --attention-dropout $ATT_DROP_OUT \
#     --warmup-updates $WARMUP_UPDATES  \
#     --clip-norm $CLIP_NORM \
#     --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
#     --lr $LR --lr-scheduler $LR_SHCE --weight-decay 0.01 \
#     --update-freq $UPDATE_FRE \
#     --skip-invalid-size-inputs-valid-test \
#     --reset-lr-scheduler \
#     --save-interval $SAVE_INTERVAL $MORE_PARA
}

distributed_train() {
     python3 -m torch.distributed.launch --nproc_per_node=$GPU_COUNT \
     --nnodes=$WORKER_COUNT --node_rank=$WORKER_ID --master_addr=$MASTER_ADDR \
     --master_port=$MASTER_PORT \
     $(which fairseq-train) $PROCESSED_DIR \
    --save-dir $MODEL_DIR \
    --user-dir /opt/tiger/fairseq/src/src_syngec/syngec_model/ \
    --bart-model-file-from-transformers bart-large-chinese \
    --task syntax-enhanced-translation \
    --arch syntax_enhanced_bart_large \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens $BATCH \
    --optimizer adam \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --lr $LR \
    --warmup-updates $WARMUP_UPDATES \
    -s src \
    -t tgt \
    --lr-scheduler polynomial_decay \
    --clip-norm 1.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch $EPOCH \
    --share-all-embeddings \
    --adam-betas '(0.9,0.999)' \
    --log-format tqdm \
    --find-unused-parameters \
    --fp16 \
    --seed $seed $MORE_PARA



  if [ $WORKER_ID -eq 0 ];then
    hadoop fs -mkdir $HDFSHOME/UGEC/$HDFS_MODEL_DIR
    hadoop fs -put -f $MODEL_DIR/checkpoint*.pt $HDFSHOME/UGEC/$HDFS_MODEL_DIR/
    echo "Upload checkpoint_best.pt to $HDFSHOME/UGEC/$HDFS_MODEL_DIR"
  fi
}

#$MORE_PARA=--reset-optimizer --reset-dataloader --reset-meters


if [ "$DO_TRAIN" = "True" ]; then
  if [ $WORKER_COUNT -eq 1 ];then
    echo "Start Single Node Training..."
    train
  else
    echo "Start $WORKER_COUNT Nodes Training..."
    distributed_train
  fi
fi

