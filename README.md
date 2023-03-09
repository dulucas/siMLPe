# siMLPe
**Back to MLP: A Simple Baseline for Human Motion Prediction(WACV 2023)** 

A simple-yet-effective network achieving **SOTA** performance.

In this paper, we propose a naive MLP-based network for human motion prediction. The network consists of only FCs, LayerNorms and Transpose operation. There is no non-linear activation in our network.

[paper link](https://arxiv.org/abs/2207.01567)

### Network Architecture
------
![image](.github/pipeline.png)

> **_NOTE:_** In the paper, we asume that FC is realized by nn.conv1D, so we have two transpose operation before/after the spacial FC layers, see the pipeline figure. While in this repo, we use nn.Linear to implement FC, so in the code there is only one transpose after the spacial FC layer. These two implementations are equivalent. (for a input tensor (b, n, d), nn.conv1D operates on dimention n, while nn.Linear operates on dimention d).

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

[H3.6M](https://drive.google.com/file/d/15OAOUrva1S-C_BV8UgPORcwmWG2ul4Rk/view?usp=share_link)

[Original stanford link](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip) has crashed, this link is a backup.

Directory structure:
```shell script
data
|-- h36m
|   |-- S1
|   |-- S5
|   |-- S6
|   |-- ...
|   |-- S11
```

[AMASS](https://amass.is.tue.mpg.de/)

Directory structure:
```shell script
data
|-- amass
|   |-- ACCAD
|   |-- BioMotionLab_NTroje
|   |-- CMU
|   |-- ...
|   |-- Transitions_mocap
```

[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)

Directory structure: 
```shell script
data
|-- 3dpw
|   |-- sequenceFiles
|   |   |-- test
|   |   |-- train
|   |   |-- validation
```

### Training
------
#### H3.6M
```bash
cd exps/baseline_h36m/
sh run.sh
```

#### AMASS
```bash
cd exps/baseline_amass/
sh run.sh
```

## Evaluation
------
#### H3.6M
```bash
cd exps/baseline_h36m/
python test.py --model-pth your/model/path
```

#### AMASS
```bash
cd exps/baseline_amass/
#Test on AMASS
python test.py --model-pth your/model/path 
#Test on 3DPW
python test_3dpw.py --model-pth your/model/path 
```

### Citation
If you find this project useful in your research, please consider cite:
```bash
@article{guo2022back,
  title={Back to MLP: A Simple Baseline for Human Motion Prediction},
  author={Guo, Wen and Du, Yuming and Shen, Xi and Lepetit, Vincent and Xavier, Alameda-Pineda and Francesc, Moreno-Noguer},
  journal={arXiv preprint arXiv:2207.01567},
  year={2022}
}
```

### Contact
Feel free to contact [Wen](wen.guo@inria.fr) or [Me](yuming.du@enpc.fr) or open a new issue if you have any questions.
