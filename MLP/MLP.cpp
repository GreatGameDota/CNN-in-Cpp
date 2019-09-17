#include "MLP.h"
#include "../utils/MatrixFunctions.h"
#include "Activation.h"
#include "Linear.h"
#include "../common.h"

MLP::MLP(double lr) : learningRate{lr}
{
}

void MLP::addDenseLayer(int units, int inputShape, string weight_init)
{
  vector<int> prevWeightShape;
  if (linearLayers.size() > 0)
  {
    prevWeightShape.push_back(linearLayers[linearLayers.size() - 1].getW().size());
    prevWeightShape.push_back(linearLayers[linearLayers.size() - 1].getW()[0].size());
  }
  linearLayers.push_back(LinearLayer(units, inputShape, weight_init, prevWeightShape));
  activationLayers.push_back(Activation());
}

void MLP::computeCost(vector<vector<double>> &result, vector<vector<double>> Y, vector<vector<double>> Y_hat)
{
  int m = Y[0].size();
  vector<vector<double>> res;
  vector<vector<double>> res2;
  sub(res, Y, Y_hat);
  square(res2, res);
  sum(res2, res2, 1);
  cost = 1.0 / (2 * m);
  double total = 0;
  for (int i = 0; i < Y.size(); i++)
  {
    total += res2[i][0];
  }
  cost *= total / Y.size();
  mult(res, res, -1.0 / m);
  result = res;
}

void MLP::train(int epochs, vector<vector<double>> X_train, vector<vector<double>> Y_train)
{
  for (int k = 0; k < epochs; k++)
  {
    // Forward
    for (int i = 0; i < linearLayers.size(); i++)
    {
      if (i == 0)
      {
        linearLayers[i].linearForward(X_train);
      }
      else
      {
        linearLayers[i].linearForward(activationLayers[i - 1].getA());
      }
      activationLayers[i].activationForward(linearLayers[i].getZ());
    }
    // Compute Cost
    vector<vector<double>> dA;
    computeCost(dA, Y_train, activationLayers[linearLayers.size() - 1].getA());
    // Backward
    for (int i = linearLayers.size() - 1; i >= 0; i--)
    {
      if (i == linearLayers.size() - 1)
      {
        activationLayers[i].activationBackward(dA);
      }
      else
      {
        activationLayers[i].activationBackward(linearLayers[i + 1].getDA_Prev());
      }
      linearLayers[i].linearBackward(activationLayers[i].getDZ());
    }
    // Update Weights and Biases
    for (int i = linearLayers.size() - 1; i >= 0; i--)
    {
      linearLayers[i].updateParams(learningRate);
    }
    epoch++;
  }
}

void MLP::predict(vector<vector<double>> &result, vector<vector<double>> X)
{
  int n = linearLayers.size() - 1;
  linearLayers[0].linearForward(X);
  activationLayers[0].activationForward(linearLayers[0].getZ());
  for (int i = 1; i <= n; i++)
  {
    linearLayers[i].linearForward(activationLayers[i - 1].getA());
    activationLayers[i].activationForward(linearLayers[i].getZ());
  }
  result = activationLayers[activationLayers.size() - 1].getA();
}

void MLP::softmax(vector<vector<double>> &result, vector<vector<double>> X)
{
  int row = X.size();
  int col = X[0].size();
  vector<vector<double>> res(row, vector<double>(col, 0));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      res[i][j] = exp(X[i][j]);
    }
  }
  vector<vector<double>> sumAll;
  sum(sumAll, res, 0);
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      res[i][j] /= sumAll[0][0];
    }
  }
  result = res;
}