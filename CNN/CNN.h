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
  void conv(std::vector<std::vector<std::vector<double>>> image, int label);
};

#endif