#ifndef LINEAR_H
#define LINEAR_H

#include <vector>
#include <cmath>
#include <string>

class LinearLayer
{
private:
  std::vector<std::vector<double>> W;
  std::vector<std::vector<double>> b;
  std::vector<std::vector<double>> A_prev;
  std::vector<std::vector<double>> dA_prev;
  std::vector<std::vector<double>> db;
  std::vector<std::vector<double>> dW;
  std::vector<std::vector<double>> Z;

  double randGaussian();

public:
  LinearLayer(int units, int input_shape, std::string init_type, std::vector<int> prevWeightShape);

  void linearForward(std::vector<std::vector<double>> _A_prev);
  void linearBackward(std::vector<std::vector<double>> upstream_grad);
  void updateParams(double learning_rate);

  std::vector<std::vector<double>> getW() const { return W; }
  std::vector<std::vector<double>> getB() const { return b; }
  std::vector<std::vector<double>> getA_Prev() const { return A_prev; }
  std::vector<std::vector<double>> getDA_Prev() const { return dA_prev; }
  std::vector<std::vector<double>> getDB() const { return db; }
  std::vector<std::vector<double>> getDW() const { return dW; }
  std::vector<std::vector<double>> getZ() const { return Z; }
};

#endif