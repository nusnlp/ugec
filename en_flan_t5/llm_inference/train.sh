#MODEL_NAME="/mnt/bn/llm-checkpoints/decapoda_research_llama_65b_hf"
#FRAMEWORK='clm'
#DATA_VER="clang8"
#RESULT_DIR="test"
#BATCH=1
#GRADIENT_ACCUMULATION_STEPS=4
#TRAIN_STEPS=20
#SAVE_STEPS=10
## SOURCE_PREFIX="correct spelling errors in the following sentence: {} "
#UPLOAD_HDFS="False"
#MORE_PARA="--bf16"

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

echo "Download Pretrainmodel..."
hadoop fs -test -e $HDFSHOME/pretrainmodel/$MODEL_NAME
if [ $? -eq 0 ];then
    echo "Detect $MODEL_NAME under $HDFSHOME/pretrainmodel/. Downloading to pretrainmodel/ ..."
    mkdir -p pretrainmodel
    hadoop fs -get $HDFSHOME/pretrainmodel/$MODEL_NAME pretrainmodel/
    MODEL_NAME=pretrainmodel/$MODEL_NAME
elif [ -d "${BYTENAS_HOME}/${MODEL_NAME}" ]; then
    echo "Detect $MODEL_NAME under ${BYTENAS_HOME}/."
    MODEL_NAME=${BYTENAS_HOME}/${MODEL_NAME}
else
    echo "Model $MODEL_NAME doesn't detected and will be downloaded from huggingface..."
fi

echo "Listening on ${RESULT_DIR} for Checkpoints..."
source utils/bashutils.sh
mkdir $RESULT_DIR
start_inotify $RESULT_DIR $HDFSHOME/LLM/$RESULT_DIR $UPLOAD_HDFS


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


hadoop fs -mkdir $HDFSHOME/LLM/$RESULT_DIR
hadoop fs -put -f $OUTPUT_DIR $HDFSHOME/LLM/$RESULT_DIR/

# wait for checkpoints uploading
sleep 10m