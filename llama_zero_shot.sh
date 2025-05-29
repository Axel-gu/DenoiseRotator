model_path="path to saved rotated model"
rotation_dir="path to log's dir"

mkdir -p $rotation_dir

python llama_zero_shot.py \
       --config_path $model_path \
       --distribute_model False \
       --state_dict_path $rotation_dir"/state_dict.pth" \
       | tee $rotation_dir/log.txt
