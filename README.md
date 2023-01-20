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

TODO:
* add `setup.py` to be able to `pip install -e .`
* link to colab notebook with inference examples on LOCATA, FRIDA, and DCASE 2022
* add examples that qualitatively show that this implementation is the same as the original
