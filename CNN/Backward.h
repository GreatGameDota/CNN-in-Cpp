#ifndef BACKWARD_H
#define BACKWARD_H

#include <vector>

void convolutionBackward(std::vector<std::vector<std::vector<double>>> &result, std::vector<std::vector<std::vector<std::vector<double>>>> &df, std::vector<std::vector<double>> &db, std::vector<std::vector<std::vector<double>>> dconv_prev, std::vector<std::vector<std::vector<double>>> conv_in, std::vector<std::vector<std::vector<std::vector<double>>>> filter, int stride = 1);
void maxpoolBackward(std::vector<std::vector<std::vector<double>>> &result, std::vector<std::vector<std::vector<double>>> dpool, std::vector<std::vector<std::vector<double>>> orig, int size = 2, int stride = 2);

#endif