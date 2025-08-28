# TemporalEval-MM

This repository contains the code for the thesis paper called _Evaluating Temporal Understanding in Video Large Multimodal Models through Action Order Comparison in Paired Videos_. The code is based on the implementation from the [_ConVis-Bench_](https://anonymous.4open.science/r/convis-E7FB/README.md) which is the currently under anonymous submission to NeurIPS 2025.

The main goal of the work is to assess temporal understanding of the Visual Large Multimodal Models (VLMMs) through the video comparison of the action order.

### Get Started

Clone this repository:

```
git clone https://github.com/annedadaa/TemporalEval-MM.git
```

Navigate to the project directory:
```
cd TemporalEval-MM
```
### Environment Setup
This and the following instuction sections are the same as of the _ConVis-Bench_ from their [anonymous repository](https://anonymous.4open.science/r/convis-E7FB/README.md). 

Create a conda environment and install all dependencies:
```
conda create -n convisenv python=3.12.2
conda activate convisenv
pip install -r requirements.txt
```

### Download Dataset
The dataset is downloaded from huggingface and saved to the provided in _--local-dir_ folder:

```
mkdir convisbench/ 
huggingface-cli download submission1335/ConViS-Bench --repo-type dataset --local-dir convisbench/
```

### Experimental Setup

Before running the experiments, you have to configure all the experimental parameters. This should be done in the [config.json](https://github.com/annedadaa/TemporalEval-MM/blob/main/utils/config.json) file.

#### Available Models
The current implementation supports three models:
1. LLaVA-OneVision-Qwen2-7B
2. Qwen2.5-VL-7B-Instruct
3. InternVL3-8b

Please note that you have to specify one of these exact model names in the configuration, otherwise you will encouter an error.

#### Pipelines

Both pipelines for the video similarity and correlation scores computation are in the [baseline](https://github.com/annedadaa/TemporalEval-MM/tree/main/baselines) directory. You have to run the code from the _project_ directory.

### Calculate Video Similarities

To run the similarities computation script on GPU, you can specify the exact GPU cards' IDs:
```
CUDA_VISIBLE_DEVICES=0,1 python -m baselines.compute_conditioned_similarities
```
Outputs will be collected and stored in the _/model_outputs_ folder, in the respective subfolder given the experimental configuration. The output examples are presented in the [output folder](https://github.com/annedadaa/TemporalEval-MM/tree/main/model_outputs/computed_conditioned_similarities/order_of_actions).

Additionally, you can enable the visualization of the frames in the configuration file and they will be stored in the _/visualizations_ folder.

### Compute Correlation Scores

When the similirities are computed, you can calculate the correlation scores between the model's outputs and humans' annotations:
```
python -m baselines.compute_conditioned_correlation
```

