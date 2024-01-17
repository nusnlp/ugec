start_inotify(){
    local local_dir=$1
    local remote_dir=$2
    hadoop fs -mkdir ${remote_dir}
    (inotifywait -m ${local_dir} -e close_write -e create -e moved_to |
        while read path action file; do
            if [[ "$file" == checkpoint-* ]]; then
                # wait for writing checkpoint
                wait=1
                while [ $wait -eq 1 ];do
                    if compgen -G "${local_dir}/$file/*.pth" > /dev/null; then
                        wait=0
                    fi
                    sleep 10s
                done

                echo "| inotifywait --> | checkpoint detected: $file"
                checkpoint_detect_handler ${local_dir}/${file} ${remote_dir}/${file} $3 &
            fi
        done) &
}

# 检测到 checkpoint 会：
# 1. 删除 deepspeed 的相关checkpoints以节省存储；
# 2. 上传 checkpoint 到 HDFS (if set 'True')；


checkpoint_detect_handler(){
    local local_ckpt=$1
    local remote_ckpt=$2

    echo "| inotifywait --> | removing DeepSpeed checkpoints to save disk storage"
    rm -r ${local_ckpt}/global_step*
    rm ${local_ckpt}/*.pth

    if [ "$3" = "True" ];then
        echo "| inotifywait --> | uploading checkpoint"
        hadoop fs -mkdir ${remote_ckpt}
        hadoop fs -put -f ${local_ckpt}/* ${remote_ckpt}/ 
        echo "| inotifywait --> | checkpoint uploaded: ${local_ckpt}"
        rm -rf ${local_ckpt}/*
    fi
}