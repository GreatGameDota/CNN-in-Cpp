#ifndef FORWARD_H
#define FORWARD_H

#include <vector>
#include <cmath>

void convolution(std::vector<std::vector<std::vector<double>>> &result, std::vector<std::vector<std::vector<double>>> image, std::vector<std::vector<std::vector<std::vector<double>>>> filter, std::vector<std::vector<double>> bias, int stride = 1);
void maxpool();
void softmax();
void categoricalCrossEntropy();

#endif