#ifndef FORWARD_H
#define FORWARD_H

#include <vector>

void convolution(std::vector<std::vector<std::vector<double>>> &result, std::vector<std::vector<std::vector<double>>> image, std::vector<std::vector<std::vector<std::vector<double>>>> filter, std::vector<std::vector<double>> bias, int stride = 1);
void ReLU(std::vector<std::vector<std::vector<double>>> &result);
void ReLU2D(std::vector<std::vector<double>> &result);
void maxpool(std::vector<std::vector<std::vector<double>>> &result, std::vector<std::vector<std::vector<double>>> image, int size = 2, int stride = 2);
void softmax(std::vector<std::vector<double>> &result, std::vector<std::vector<double>> X);
void categoricalCrossEntropy(double &result, std::vector<std::vector<double>> probs, std::vector<std::vector<double>> label);

#endif