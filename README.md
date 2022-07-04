# siMLPe
Official implementation for our paper **Back to MLP: A Simple Baseline for Human Motion Prediction**

[paper link](https://arxiv.org/abs/2002.12730)

### Network Architecture
------
<p align="center">
<img src="https://github.com/dulucas/siMLPe/blob/main/.github/pipeline_v15.png" width="200" height="400">
</p>

### Requirements
------
- PyTorch >= 1.5
- Numpy
- CUDA >= 10.1
- Easydict
- pickle
- einops
- scipy
- six

### Data Preparation
------
Download all the data and put them in the `./data` directory.

[H3.6M](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip)

Directory structure:
```shell script
H3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```

[AMASS](https://amass.is.tue.mpg.de/)

Directory structure:
```shell script
amass
|-- ACCAD
|-- BioMotionLab_NTroje
|-- CMU
|-- ...
`-- Transitions_mocap
```

[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)

Directory structure: 
```shell script
3dpw
|-- sequenceFiles
|   |-- test
|   |-- train
|   |-- validation
```

### Training
------
#### H3.6M
```
cd exps/baseline_h36m/
sh run.sh
```

#### AMASS
```
cd exps/baseline_amass/
sh run.sh
```

## Evaluation
------
#### H3.6M
```
cd exps/baseline_h36m/
python test.py --model-pth your/model/path
```

#### AMASS
```
cd exps/baseline_amass/
# test on AMASS
python test.py --model-pth your/model/path 
# test on 3DPW
python test_3dpw.py --model-pth your/model/path 
```
