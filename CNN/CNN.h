#ifndef CNN_H
#define CNN_H

#include <vector>
#include <cmath>

class CNN
{
private:
  double randGaussian();

public:
  void train();
  void adamGD();
  void conv(std::vector<std::vector<std::vector<std::vector<double>>>> &_df1, std::vector<std::vector<std::vector<std::vector<double>>>> &_df2, std::vector<std::vector<double>> &_dw3, std::vector<std::vector<double>> &_dw4, std::vector<std::vector<double>> &_db1, std::vector<std::vector<double>> &_db2, std::vector<std::vector<double>> &_db3, std::vector<std::vector<double>> &_db4, std::vector<std::vector<std::vector<double>>> image, std::vector<std::vector<double>> label);
};

#endif