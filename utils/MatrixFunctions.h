#ifndef MATRIX_FUNCTIONS_H
#define MATRIX_FUNCTIONS_H

#include <vector>

void dot(std::vector<std::vector<double>> &result, std::vector<std::vector<double>> matrixA, std::vector<std::vector<double>> matrixB);
void transpose(std::vector<std::vector<double>> &result, std::vector<std::vector<double>> matrix);
void sub(std::vector<std::vector<double>> &result, std::vector<std::vector<double>> matrixA, std::vector<std::vector<double>> matrixB);
void mult(std::vector<std::vector<double>> &result, std::vector<std::vector<double>> matrix, double n);
void multMatrices(std::vector<std::vector<double>> &result, std::vector<std::vector<double>> matrixA, std::vector<std::vector<double>> matrixB);
void add(std::vector<std::vector<double>> &result, std::vector<std::vector<double>> matrixA, std::vector<std::vector<double>> matrixB);
void sum(std::vector<std::vector<double>> &result, std::vector<std::vector<double>> matrix, int axis);
void square(std::vector<std::vector<double>> &result, std::vector<std::vector<double>> matrix);

#endif