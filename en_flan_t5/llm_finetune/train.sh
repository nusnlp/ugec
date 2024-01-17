MODEL_NAME="pretrain" ## the folder name which store previous checkpoint
FRAMEWORK='seq2seq' 
DATA_VER="train" ## train represents the downloaded train.jsonl
RESULT_DIR="stage3" ## resulting ckpt folder
BATCH=256
GRADIENT_ACCUMULATION_STEPS=1
TRAIN_STEPS=1000000
SAVE_STEPS=100
MORE_PARA="--bf16 --gradient_checkpointing"
if [ -z "$MODEL_NAME" ]
then
    echo "MODEL_NAME is None"
    exit 0
fi
if [ -z "$FRAMEWORK" ]
then
    echo "FRAMEWORK is None and please choose from ['seq2seq', 'clm']"
    exit 0
fi
if [ -z "$DATA_VER" ]
then
    echo "DATA_VER is None"
    exit 0
fi
if [ -z "$RESULT_DIR" ]
then
    echo "RESULT_DIR is None"
    exit 0
fi

if [ -z "$EPOCH" ]
then
    EPOCH=10
fi
if [ -z "$BATCH" ]
then
    BATCH=512
fi
if [ -z "$GRADIENT_ACCUMULATION_STEPS" ]
then
    GRADIENT_ACCUMULATION_STEPS=4
fi
if [ -z "$SAVE_STEPS" ]
then
    SAVE_STEPS=500
fi
if [ -z "$SOURCE_PREFIX" ]
then
    SOURCE_PREFIX="{} "
fi
if [ -z "$UPLOAD_HDFS" ]
then
    UPLOAD_HDFS="False"
fi
if [ -z "$MORE_PARA" ]
then
    MORE_PARA=""
fi


echo "FRAMEWORK=$FRAMEWORK"
echo "MODEL_NAME=$MODEL_NAME"
echo "DATA_VER=$DATA_VER"
echo "RESULT_DIR=$RESULT_DIR"
echo "BATCH=$BATCH"
echo "GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
echo "TRAIN_STEPS=$TRAIN_STEPS"
echo "SAVE_STEPS=$SAVE_STEPS"
echo "SOURCE_PREFIX=$SOURCE_PREFIX"
echo "UPLOAD_HDFS=$UPLOAD_HDFS"
echo "MORE_PARA=$MORE_PARA"

export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_SOCKET_IFNAME=eth0,eth2
export NCCL_IB_DISABLE=0
export NCCL_BLOCKING_WAIT=1


echo "Download Data..."
mkdir -p data
hadoop fs -get $HDFSHOME/data/data_${DATA_VER} data/
DATA_DIR=data/data_${DATA_VER}


sleep 10s 

train_args=(
    --do_train
    # [model]
    --framework $FRAMEWORK
    --model_name_or_path $MODEL_NAME
    --source_prefix "$SOURCE_PREFIX"
    # [optimizer]
    --max_steps $TRAIN_STEPS
    --per_device_train_batch_size $BATCH 
    --per_device_eval_batch_size $BATCH
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS
    --train_file $DATA_DIR/train.json --validation_file $DATA_DIR/validation.json
    --lr_scheduler_type linear
    # [deepspeed]
    # --deepspeed ds_config_zero3.json 
    # [checkpoint]
    --output_dir $RESULT_DIR 
    --save_steps $SAVE_STEPS
    --overwrite_output_dir
    # [log]
    --report_to wandb --logging_steps 1
    --run_name $RESULT_DIR
    # [other]
    ${MORE_PARA}
)

echo "Training..."
NCCL_P2P_DISABLE=1 python3 -u -m torch.distributed.run \
    --nproc_per_node $ARNOLD_WORKER_GPU \
    --nnodes $ARNOLD_WORKER_NUM \
    --rdzv_endpoint $ARNOLD_WORKER_0_HOST:$ARNOLD_WORKER_0_PORT \
    --rdzv_backend c10d \
    --rdzv_conf read_timeout=31600 \
    finetune.py "${train_args[@]}"

sleep 10m