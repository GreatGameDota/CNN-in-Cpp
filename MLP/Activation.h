#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include <cmath>

class Activation
{
private:
  std::vector<std::vector<double>> A;
  std::vector<std::vector<double>> dZ;

  double sigmoid(double num) { return 1 / (1 + std::exp(-1 * num)); }
  double dSigmoid(double num) { return num * (1 - num); }

public:
  void activationForward(std::vector<std::vector<double>> Z);
  void activationBackward(std::vector<std::vector<double>> upstream_grad);

  std::vector<std::vector<double>> getA() const { return A; }
  std::vector<std::vector<double>> getDZ() const { return dZ; }
};

#endif