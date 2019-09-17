#ifndef MLP_H
#define MLP_H

#include <vector>
#include <cmath>
#include <string>
#include "Linear.h"
#include "Activation.h"

class MLP
{
private:
  double learningRate{0};
  double cost{1};
  std::vector<double> costs;
  int epoch{0};
  std::vector<LinearLayer> linearLayers;
  std::vector<Activation> activationLayers;

public:
  MLP(double lr);

  void addDenseLayer(int units, int inputShape, std::string weight_init);
  void computeCost(std::vector<std::vector<double>> &result, std::vector<std::vector<double>> Y, std::vector<std::vector<double>> Y_hat);
  void train(int epochs, std::vector<std::vector<double>> X_train, std::vector<std::vector<double>> Y_train);
  void predict(std::vector<std::vector<double>> &result, std::vector<std::vector<double>> X);
  void softmax(std::vector<std::vector<double>> &result, std::vector<std::vector<double>> X);

  double getLR() const
  {
    return learningRate;
  }
  double getCost() const { return cost; }
  std::vector<double> getCosts() const { return costs; }
  int getEpoch() const { return epoch; }
  std::vector<LinearLayer> getLinearLayers() const { return linearLayers; }
  std::vector<Activation> getActivationLayers() const { return activationLayers; }
};

#endif