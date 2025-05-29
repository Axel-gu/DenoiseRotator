model_path="path to Llama-3-8B"
rotation_epoch=2000
rotation_dir="path to log's dir"

mkdir -p $rotation_dir

python qwen_prune.py $model_path c4 \
       --sparsity .5 \
       --rotation_dir $rotation_dir \
       --rotation_epoch $rotation_epoch \
       --method wanda \
       --num_batch 1 \
       --true-sequential \
       --distribute_model True
    #    | tee $rotation_dir/log.txt
