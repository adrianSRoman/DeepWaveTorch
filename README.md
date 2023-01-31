## DeepWave: A Recurrent Neural-Network for Real-Time Acoustic Imaging (PyTorch)

<img src="https://github.com/adrianSRoman/DeepWaveTorch/blob/master/figures/task4_recording1.gif" width="900" height="320"/>

This repository contains a PyTorch implementation of the DeepWave model originally published at NeurIPS 2019

| [paper](https://proceedings.neurips.cc/paper/2019/file/e9bf14a419d77534105016f5ec122d62-Paper.pdf) | [original code](https://github.com/imagingofthings/DeepWave) |

Get started with DeepWave (PyTorch) inference

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BC72KmoAyeydS0X7Dti3fFxzLAxfbdWC?usp=sharing)

## Main building blocks:

- SphericalChebConv: Spherical Chebyshev graph convolutions
- BackProjLayer: project correlation matrix into image form (intensity map form)
- ReTanh: Rectified hyperbolic tangent action function 
- DeepWave: the actual model architecture

DeepWave (PyTorch) architecture:
```
Deepwave: input=S (visibility matrix), trainable={mu, D, tau}
    y <- BackProjLayer(S) 
conv4 <- SphericalChebConv(I_init) + y
conv4 <- ReTanh(conv4)
conv3 <- SphericalChebConv(conv4) + y
conv3 <- ReTanh(conv3)
conv2 <- SphericalChebConv(conv3) + y
conv2 <- ReTanh(conv2)
conv1 <- SphericalChebConv(conv2) + y
conv1 <- ReTanh(conv1)
conv0 <- SphericalChebConv(conv1) + y
I_out <- ReTanh(conv0)
```

Many of the operations used in this implementation were borrowed from the repository [deepsphere-pytorch](https://github.com/deepsphere/deepsphere-pytorch).

## Installation

#### First time setup
Create a Python virtual environment
```
python3 -m venv /path/to/your_new_venv
```

Start `your_new_venv`
```
source /path/to/<your_new_venv>/bin/activate
```

- Clone `DeepWaveTorch` (this repo!).
```
git clone git@github.com:adrianSRoman/DeepWaveTorch.git 
```

- Initialize submodules
```
git submodule init
```

- Start `ImoT_tools`: plotting library used to nicely visualize DeepWave's output
```
cd ImoT_tools
pip install -r requirements.txt
python3 setup.py develop
```

- Start `DeepWave`: original DeepWave implementation. Used for benchmarking against the PyTorch implementation. Data loaders are also re-used from the original implementation. 
```
cd DeepWave
python3 setup.py develop
```

- Start `DeepWaveTorch`: new DeepWave PyTorch implementation.
```   
cd DeepWaveTorch
pip install -r requirements.txt
python3 setup.py develop
```

### Executing example `notebooks`

- Start `your_new_venv`
```
source /path/to/<your_new_venv>/bin/activate
```

- Create a Jupyter Kernel to contain your required packages (first time setup only)
```
pip install ipykernel
ipython kernel install --user --name=<your_new_kernel_name>
```

- Start Jupyter

```
jupyter notebook
```

- Select a given notebook you want to work with
- Select `your_new_kernel_name` under: Kernel > Change kernel > `your_new_kernel_name`

## Extracting visibility matrices (DeepWave's input)

Extracting visibility matrices `S` is perhaps the main aspect you will need because they are DeepWave's input! To save you some time we created a simple notebook that for a given audio track of M microphones and N samples (we use an eigenmike32), it generates a visibility matrix (MxM) over 100 msec audio frames. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BC72KmoAyeydS0X7Dti3fFxzLAxfbdWC?usp=sharing)

Note: the data extraction for the experiments in this repo (including extracting visibility matrices) was done using the `run.sh` files from the original DeepWave code [see example run.sh](https://github.com/imagingofthings/DeepWave/blob/master/datasets/FRIDA/run.sh). The `run.sh` files perform three tasks: 

(1) Extract data from a mic array of N channels to generate visibility matrices `S` (this is what DeepWave takes as input). Extract a ground truth intensity field `I` which DeepWave learns how to generate. The script will then generate a `.npz` file.

(2) Merge datasets in single `.npz` files for the 9 different extracted frequency bands across all tracks.

(3) Train the original DeepWave model. After training is done, you will find `.npz` files containing the trained weights of the model for each frequency band.

Overall, you can either use the colab notebook to extract visibility matrices, or execute the `run.sh` scripts from the original DeepWave repo (you choose).

## Qualitative and quantitative comparison against the original DeepWave implementation

#### Qualitatively, the implementation from this repository generates the same intensity fields as the original DeepWave implementation.

##### Inferred intensity field for a single frequency band: DeepWave original Vs. DeepWave PyTorch

<p align="center" width="100%">
    <img width="60%" src="https://github.com/adrianSRoman/DeepWaveTorch/blob/master/figures/DeepWave_fields_comparison.png">
</p>


## YouTube presentation

[![Mexico LatAm BISH Bash: DeepWave](https://img.youtube.com/vi/ZO5jfqY_NwA/0.jpg)](https://www.youtube.com/watch?v=ZO5jfqY_NwA)


## License
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
