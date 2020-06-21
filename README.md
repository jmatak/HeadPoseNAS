# HeadPoseNAS
[Master's thesis] Transfer learning for Head pose estimation, enhanced by neural architecture search. 

**Author: Josip Matak**

### Image demo

| Example 1 | Example 2 |
| --- | --- |
| <img src="https://github.com/jmatak/HeadPoseNAS/blob/master/images/example_0.png" height="220"/> |  <img src="https://github.com/jmatak/HeadPoseNAS/blob/master/images/example_1.png" height="220"/> |


| Example 3 | Example 4 |
| --- | --- |
| <img src="https://github.com/jmatak/HeadPoseNAS/blob/master/images/example_2.png" height="220"/> |  <img src="https://github.com/jmatak/HeadPoseNAS/blob/master/images/example_3.png" height="220"/> |

## Abstract
Estimating head pose is a useful practice in many industries, especially those of a medi-cal, technical, or automotive nature. Precision and speed of execution are the key things thatmake these methods useful in real-world practice.The solution to this task is offered by the current hot topic in deep learning, a machinelearning subset.  The main contribution of this paper is the transfer learning technique andseveral different search methods for the best neural network architecture, built from prede-fined blocks. The results presented in the paper suggest that features extracted from the facerecognition method are sufficient for estimating head pose.  Also, deep model architectureoptimization methods have shown improvements in execution speed and precision by opti-mizing the multi-criterion objective function.  Although this is a task that requires time andmemory resources, automated search allows users to change the main focus of research intodifferent problems in the same domain.

## Platform
+ Tensorflow 1.14, Numpy, OpenCV 
+ NVIDIA GeForce RTX 2070 SUPER
+ Ubuntu 18.04

## Dependencies
```
absl-py==0.9.0
astor==0.8.1
cma==3.0.3
gast==0.3.3
google-pasta==0.2.0
grpcio==1.29.0
h5py==2.10.0
importlib-metadata==1.6.1
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.2
Markdown==3.2.2
numpy==1.19.0
opencv-python==4.2.0.34
pandas==1.0.5
protobuf==3.12.2
Py-BOBYQA==1.2
pyswarm==0.6
python-dateutil==2.8.1
pytz==2020.1
scipy==1.5.0
six==1.15.0
tensorboard==1.14.0
tensorflow-estimator==1.14.0
tensorflow-gpu==1.14.0
termcolor==1.1.0
tqdm==4.46.1
Werkzeug==1.0.1
wrapt==1.12.1
zipp==3.1.0
```

## Reproducing results

### 1. Data pre-processing

Ensure to download datasets below and put them in `dataset` folder

+ [300W-LP, AFLW2000](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
+ [BIWI](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)

Unzip the files, run `extract_pose.py` for each folder, and  `prepare_dataset.py` for each extracted numpy saved array. Then, run `split_dataset.py` to split training dataset 30-70.
#### For lazy

Or if you are lazy enough, run `loader.sh`.


### 2. Training and testing
To train and generate models with evolutionary algorithms follow next procedure

```
python3 search.py --alg cmaes|genetic|bobyqa|pso|random
```
Depending on the wanted algorithm
