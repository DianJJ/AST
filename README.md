# AST
Analysis-by-Synthesis Transformer for Single-View 3D Reconstruction 

### Dependencies :wrench:

```
conda env create -f environment.yml
conda activate ast
```

### Dataset :arrow_down:

```
bash scripts/download_data.sh
```

### Train models from scratch

To launch a training from scratch, run:

```
cuda=gpu_id config=filename.yml tag=run_tag ./scripts/pipeline.sh
```

where `gpu_id` is a device id, `filename.yml` is a config in `configs` folder, `run_tag` is a tag for the experiment.

Results are saved at `runs/${DATASET}/${DATE}_${run_tag}` where `DATASET` is the dataset name 
specified in `filename.yml` and `DATE` is the current date in `mmdd` format.

## Contact
djia7@uic.edu

## Citation
If you find this repo useful for your research, please consider citing our paper:

```bibtex
@inproceedings{jia2024ast,
  title={Analysis-by-Synthesis Transformer for Single-View 3D Reconstruction },
  author={Jia, Dian and Ruan, Xiaoqian and Xia, Kun and Zou, Zhiming and Wang, Le and Tang, Wei},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```
## Acknowledgement

Many thanks to these excellent open source projects:
- [Unicorn](https://github.com/monniert/unicorn)
- [cmr](https://github.com/akanazawa/cmr)
- [SoftRas](https://github.com/ShichenLiu/SoftRas)
- [detr](https://github.com/facebookresearch/detr)
