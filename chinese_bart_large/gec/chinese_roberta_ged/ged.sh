input=$1
python3 roberta_ged.py \
  --weight_path "ckpt" \
  --checkpoint "checkpoint_best.pt" \
  --input_data "$input" \
  --output_file "$input.label"