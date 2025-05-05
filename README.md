# Adaptive Representation Construction from CLIP Embeddings for Test-Time Adaptation 
[![Website](https://kdiaaa.github.io/tda/)

Official implementation of the paper: "Efficient Test-Time Adaptation of Vision-Language Models" [CVPR 2024].

## Requirements 
### Installation
Follow these steps to set up a conda environment and ensure all necessary packages are installed:

```bash
git clone https://github.com/kdiAAA/TDA.git
cd RFIA

conda create -n tda python=3.7
conda activate rfia

# The results are produced with PyTorch 1.12.1 and CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
```

### Dataset
To set up all required datasets, kindly refer to the guidance in [DATASETS.md](docs/DATASETS.md), which incorporates steps for two benchmarks.

## Run TDA
### Configs
The configuration for TDA hyperparameters in `configs/dataset.yaml` can be tailored within the provided file to meet the needs of various datasets. This customization includes settings for both the positive and negative caches as outlined below:
* **Cache Configuration:** Adjustments can be made to the `shot_capacity`, `alpha`, and `beta` values to optimize performance.

For ease of reference, the configurations provided aim to achieve optimal performance across datasets on two benchmarks, consistent with the results documented in our paper. Adjusting parameters like `alpha` and `beta` within the positive cache lets you fine-tune things to match the unique needs of each dataset.

### Running
To execute the RFIA, navigate to the `scripts` directory, where you'll find 4 bash scripts available. Each script is designed to apply the method to two benchmarks, utilizing either the ResNet50 or ViT/B-16 as the backbone architecture. The scripts process the datasets sequentially, as indicated by the order divided by '/' in the script. WandB logging is activated by default. If you wish to deactivate this feature, simply omit the `--wandb-log` argument. 

Below are instructions for running TDA on both Out-of-Distribution (OOD) and Cross-Domain benchmarks using various backbone architectures. Follow the steps suited to your specific needs:"

#### OOD Benchmark
* **ResNet50**: Run TDA on the OOD Benchmark using the ResNet50 model:
```
bash ./scripts/run_ood_benchmark_rn50.sh 
```
* **ViT/B-16**: Run TDA on the OOD Benchmark using the ViT/B-16 model.
```
bash ./scripts/run_ood_benchmark_vit.sh 
```

#### Cross-Domain Benchmark
* **ResNet50**: Run TDA on the Cross-Domain Benchmark using the ResNet50 model:
```
bash ./scripts/run_cd_benchmark_rn50.sh 
```
* **ViT/B-16**: Run TDA on the Cross-Domain Benchmark using the ViT/B-16 model.
```
bash ./scripts/run_cd_benchmark_vit.sh 
```

### All Benchmark
* **All**: All scripts can be run simultaneously.
```
bash ./scripts/run_all_scripts.sh 
```

## Contact
If you have any questions, feel free to create an issue in this repository or contact us via email at hyewook@inha.edu.

## Acknowledgements
Our gratitude goes to the authors of [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter), [TPT](https://github.com/azshue/TPT), and [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp) for sharing their work through open-source implementation and for providing detailed instructions on data preparation.

### Thanks to TDA
Our code is based on the code in the TDA(https://kdiaaa.github.io/tda). We appreciate their efforts.