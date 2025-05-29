# DenoiseRotator

This repository contains the code associated with the paper "DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration".

## Dependencies

To run the code in this repository, the following Python packages are needed:

- `transformers`: tested on v4.50.3
- `datasets`: tested on v3.5.0
- `lm_eval`
- `torch`: tested on v2.3.1

## Usage

### Rotate and Prune using DenoiseRotator

The following example command integrate orthogonal matrices and prune Llama-3-8B:

```bash
model_path="path to Llama-3-8B"
rotation_epoch=2000
rotation_dir="path to log's dir"

mkdir -p $rotation_dir

python llama_prune.py $model_path c4 \
       --sparsity .5 \
       --rotation_dir $rotation_dir \
       --rotation_epoch $rotation_epoch \
       --method sparsegpt \
       --num_batch 1 \
       --true-sequential \
       --distribute_model False \
       | tee $rotation_dir/log.txt
```

This repository currently supports pruning methods such as `sparsegpt`, `wanda`, and `magnitude` pruning. Please ensure you specify the desired method using the `--method` option. If you wish to use other pruning method, please add corresponding codes in `sparsegpt.py`.

### Evaluating the Rotated Model

To evaluate a model after rotation and pruning, use the following command. For large models that cannot fit into a single GPU, distributed evaluation is recommended:

```bash
model_path="path to saved rotated model"
rotation_dir="path to log's dir"

mkdir -p $rotation_dir

python llama_zero_shot.py \
       --config_path $model_path \
       --distribute_model True \
       --state_dict_path $rotation_dir"/state_dict.pth" \
       | tee $rotation_dir/log.txt
```

### Custom Model Pruning

If you wish to prune other models, please create corresponding scripts named `xxx_prune.py` and `xxx_utils.py`. This repository includes implementations for `llama`, `qwen2.5`, and `mistral`.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{denoiserotator,
  title={DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration},
  author={},
  year={2025},
  journal={}
}
```

## Acknowledgments

The code in this repository is inspired by the official implementations of SparseGPT, WANDA, and SliceGPT. We express our gratitude to these projects for their valuable contributions to the field.

## License

This project is licensed under the [Apache-2.0] license. See the LICENSE file for more details.



