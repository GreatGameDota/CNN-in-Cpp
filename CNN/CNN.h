#ifndef CNN_H
#define CNN_H

#include <vector>
#include <cmath>

class CNN
{
private:
  

public:
  void train();
  void adamGD();
  void conv();

  void test(std::vector<std::vector<std::vector<double>>> image);
};

#endif