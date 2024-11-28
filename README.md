# AST
Analysis-by-Synthesis Transformer for Single-View 3D Reconstruction 
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
