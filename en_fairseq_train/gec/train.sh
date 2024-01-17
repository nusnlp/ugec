MODEL_ARCH=bart_base
cpu_num=88
SAVE_INTERVAL=1
seed=42
ARR_DROP_OUT=0.1
RESTORE_MODEL=$1 ## path to the checkpoint to be loaded
DROP_OUT=0.3
DATA_VER=test ## path to the data-bin file
WARMUP_UPDATES=2000
CLIP_NORM=0.1
LR=3e-5
MORE_PARA='--find-unused-parameters --label-smoothing 0.1 --total-num-update 1000000000 --validate-interval-updates 100 --source-lang src --target-lang tgt --save-interval-updates 100 --keep-interval-updates 2'
HDFS_MODEL_DIR=$3 ## save path
BATCH=10000
EPOCH=20
CRITERION=label_smoothed_cross_entropy
PROCESSED_DIR=$2 ## path to the data-bin file
MODEL_DIR=$HDFS_MODEL_DIR
UPDATE_FRE=1
sudo pip3 uninstall -y numpy
if [ -z "$DATA_VER" ]
then
    echo "DATA_VER is None"
    exit 0
fi


if [ -z "$MODEL_ARCH" ]
then
    echo "MODEL_ARCH is None"
    exit 0
fi

if [ -z "$HDFS_MODEL_DIR" ]
then
    echo "HDFS_MODEL_DIR is None"
    exit 0
fi

if [ -z "$Lang" ]
then
    Lang=en
fi


if [ -z "$BPE" ]
then
    BPE=none

fi


if [ -z "$cpu_num" ]
then
    cpu_num=4

fi


if [ -z "$FAIRSEQ" ]
then
    FAIRSEQ=/opt/tiger/fairseq
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
# hadoop fs -get $HDFSHOME/${RESTORE_MODEL}
#  hadoop fs -get $HDFSHOME/PLM/gpt2 PLM/gpt2


# PROCESSED_DIR=process_${DATA_VER}_${MODEL_ARCH}_${Lang}
# mkdir -p $PROCESSED_DIR
# hadoop fs -get $HDFSHOME/data/UGEC/en/BART/$HDFS_PROCESSED_DIR
#hadoop fs -get $HDFSHOME/data/UGEC/en/BART/LM-Critic-Pretrain-GED-bin


echo "PROCESSED_DIR:$PROCESSED_DIR"
echo "RESTORE_MODEL:$RESTORE_MODEL"
echo "MODEL_DIR:$MODEL_DIR"
echo "MORE_PARA:$MORE_PARA"


train() {
  fairseq-train $PROCESSED_DIR \
     --save-dir $MODEL_DIR \
     --restore-file $RESTORE_LOCAL_MODEL \
     --task $TASK \
     --layernorm-embedding --share-all-embeddings \
     --share-decoder-input-output-embed \
     --required-batch-size-multiple 1 \
     --arch $MODEL_ARCH \
     --criterion $CRITERION \
     --max-epoch $EPOCH \
     --seed $seed \
     --log-format simple \
     --max-tokens $BATCH \
     --dropout $DROP_OUT --attention-dropout $ATT_DROP_OUT \
     --warmup-updates $WARMUP_UPDATES  \
     --clip-norm $CLIP_NORM \
     --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
     --lr $LR --lr-scheduler $LR_SHCE --weight-decay 0.01 \
     --update-freq $UPDATE_FRE \
     --skip-invalid-size-inputs-valid-test \
     --reset-lr-scheduler \
     --save-interval $SAVE_INTERVAL $MORE_PARA

   # hadoop fs -mkdir $HDFSHOME/UGEC/$HDFS_MODEL_DIR
   # hadoop fs -put -f $MODEL_DIR/checkpoint*.pt $HDFSHOME/UGEC/$HDFS_MODEL_DIR/
   # echo "Upload checkpoint_best.pt to $HDFSHOME/UGEC/$HDFS_MODEL_DIR"
}

distributed_train() {
     python3 -m torch.distributed.launch --nproc_per_node=$GPU_COUNT \
     --nnodes=$WORKER_COUNT --node_rank=$WORKER_ID --master_addr=$MASTER_ADDR \
     --master_port=$MASTER_PORT \
     $(which fairseq-train) $PROCESSED_DIR \
     --save-dir $MODEL_DIR \
     --restore-file $RESTORE_LOCAL_MODEL \
     --task $TASK \
     --layernorm-embedding --share-all-embeddings \
     --share-decoder-input-output-embed \
     --required-batch-size-multiple 1 \
     --arch $MODEL_ARCH \
     --criterion $CRITERION \
     --max-epoch $EPOCH \
     --seed $seed \
     --log-format simple \
     --max-tokens $BATCH \
     --dropout $DROP_OUT --attention-dropout $ATT_DROP_OUT \
     --warmup-updates $WARMUP_UPDATES  \
     --clip-norm $CLIP_NORM \
     --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
     --lr $LR --lr-scheduler $LR_SHCE --weight-decay 0.01 \
     --update-freq $UPDATE_FRE \
     --skip-invalid-size-inputs-valid-test \
     --reset-lr-scheduler \
     --save-interval $SAVE_INTERVAL $MORE_PARA



  # if [ $WORKER_ID -eq 0 ];then
  #   hadoop fs -mkdir $HDFSHOME/UGEC/$HDFS_MODEL_DIR
  #   hadoop fs -put -f $MODEL_DIR/checkpoint*.pt $HDFSHOME/UGEC/$HDFS_MODEL_DIR/
  #   echo "Upload checkpoint_best.pt to $HDFSHOME/UGEC/$HDFS_MODEL_DIR"
  # fi
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

