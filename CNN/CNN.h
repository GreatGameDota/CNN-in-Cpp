#ifndef CNN_H
#define CNN_H

#include <vector>

class CNN
{
private:
  double lr{0.01};
  double beta1{.95};
  double beta2{.99};
  std::vector<std::vector<int>> params;
  std::vector<std::vector<std::vector<std::vector<double>>>> f1;
  std::vector<std::vector<double>> b1;
  std::vector<std::vector<std::vector<std::vector<double>>>> f2;
  std::vector<std::vector<double>> b2;
  std::vector<std::vector<double>> w3;
  std::vector<std::vector<double>> b3;
  std::vector<std::vector<double>> w4;
  std::vector<std::vector<double>> b4;

  double randGaussian();
  void adamGD(int imageAmount, std::vector<std::vector<std::vector<std::vector<double>>>> images, std::vector<int> labels, std::vector<double> &cost);
  void conv(double &_loss, std::vector<std::vector<std::vector<std::vector<double>>>> &_df1, std::vector<std::vector<std::vector<std::vector<double>>>> &_df2, std::vector<std::vector<double>> &_dw3, std::vector<std::vector<double>> &_dw4, std::vector<std::vector<double>> &_db1, std::vector<std::vector<double>> &_db2, std::vector<std::vector<double>> &_db3, std::vector<std::vector<double>> &_db4, std::vector<std::vector<std::vector<double>>> image, std::vector<std::vector<double>> label);

public:
  CNN(std::vector<std::vector<int>> params);

  void train(int epochs, int imageAmount, std::vector<std::vector<std::vector<std::vector<double>>>> images, std::vector<int> labels);
};

#endif