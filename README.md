# Convolutional Neural Network in C++

Implemented following [Alejandro Escontrela](https://github.com/Alescontrela)'s Towards Data Science Medium Article ["Convolutional Neural Networks from the ground up"](https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1) and [Python Implementation](https://github.com/Alescontrela/Numpy-CNN) on Github  

This Network is built and trained for the MNIST data set.  

The dimensions of this model are:  
8 filters for the 2 convolution layers then 1 maxpool before a 2 layer fully connected layer.  

### Run this program

To run this program simply have MinGW C++ compiler installed and clone this repo:

```shell
> git clone <repo link>
```

Then build and run the exe:

```shell
> g++ *.cpp */*.cpp -o main
> ./main
```
