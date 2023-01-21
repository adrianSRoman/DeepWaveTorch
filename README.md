## DeepWave: A Recurrent Neural-Network for Real-Time Acoustic Imaging (PyTorch)


This repository contains a pytorch implementation of the DeepWave model originally published at Neurips 2019

| [paper](https://proceedings.neurips.cc/paper/2019/file/e9bf14a419d77534105016f5ec122d62-Paper.pdf) | [original code](https://github.com/imagingofthings/DeepWave) |

This is work in progress!

Main building blocks:
- SphericalChebConv: Spherical Chebyshev graph convolutions
- BackProjLayer: project correlation matrix into image form (intensity map form)
- ReTanh: Rectified hyperbolic tangent action function 
- DeepWave: the actual model architecture

DeepWave (PyTorch) architecture:
```
Deepwave: input=S, trainable={mu, D, tau}
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

### Installation

#### First time setup
Create a Python virtual environment
```
python3 -m venv /path/to/your_new_venv
```

Start `your_new_venv`
```
source /path/to/your_new_venv/bin/activate
```

If the `modules` subdirectories ar empty execute:
```
git submodule update --init --recursive
```

#### Installing dependencies (order matters)

- Install `ImoT_tools`: plotting library used to nicely visualize DeepWave's output
```
git clone git@github.com:imagingofthings/ImoT_tools.git
cd path/to/ImoT_tools
pip install -r requirements.txt
python3 setup.py develop
```

- Install `DeepWave`: original DeepWave NumPy implementation. Used for benchmarking against the PyTorch implementation. Data loaders are also re-used from the original implementation. 
```
git clone git@github.com:imagingofthings/DeepWave.git
cd path/to/DeepWave
python3 setup.py develop
```

- Install `DeepWaveTorch`: new DeepWave PyTorch implementation (this repo!).
```
git clone git@github.com:adrianSRoman/DeepWaveTorch.git    
cd path/to/DeepWaveTorch
python3 setup.py develop
```



