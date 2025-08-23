# TemporalEval-MM

This repository contains the code for the thesis paper called _Evaluating Temporal Understanding in Video Large Multimodal Models through Action Order Comparison in Paired Videos_. The code is based on the implementation from the [_ConVis-Bench_](https://anonymous.4open.science/r/convis-E7FB/README.md) which is the currently under anonymous submission to NeurIPS 2025.

The main goal of the work is to assess temporal understanding of the VLMMs through the video comparison of the action order.
TODO: add some results and graphs here.

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

### Calculate Video Similarities
TODO: add instructions 

### Compute Correlation Scores
TODO: add instructions 


