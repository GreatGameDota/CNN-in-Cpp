#include "Linear.h"
#include "../utils/MatrixFunctions.h"
#include "../common.h"

LinearLayer::LinearLayer(int units, int input_shape, string init_type, vector<int> prevWeightShape)
{
  srand(time(NULL));
  int n_in;
  if (input_shape != 0 && prevWeightShape.size() == 0)
  {
    n_in = input_shape;
  }
  else
  {
    n_in = prevWeightShape[0];
  }
  int n_out = units;
  vector<vector<double>> res1(n_out, vector<double>(n_in, 0));
  for (int i = 0; i < n_out; i++)
  {
    for (int j = 0; j < n_in; j++)
    {
      if (init_type == "plain")
      {
        res1[i][j] = randGaussian();
      }
      if (init_type == "xavier")
      {
        res1[i][j] = randGaussian() / sqrt(n_in);
      }
      if (init_type == "he")
      {
        res1[i][j] = randGaussian() * sqrt(2 / n_in);
      }
    }
  }
  W = res1;
  vector<vector<double>> res2;
  for (int i = 0; i < n_out; i++)
  {
    res2.push_back({0});
  }
  b = res2;
}

void LinearLayer::linearForward(vector<vector<double>> _A_prev)
{
  A_prev = _A_prev;
  vector<vector<double>> res;
  dot(res, W, _A_prev);
  add(res, res, b);
  Z = res;
}

void LinearLayer::linearBackward(vector<vector<double>> upstream_grad)
{
  vector<vector<double>> res;
  transpose(res, A_prev);
  dot(res, upstream_grad, res);
  dW = res;
  sum(res, upstream_grad, 1);
  db = res;
  transpose(res, W);
  dot(res, res, upstream_grad);
  dA_prev = res;
}

void LinearLayer::updateParams(double learning_rate)
{
  vector<vector<double>> res;
  mult(res, dW, learning_rate);
  sub(res, W, res);
  W = res;
  mult(res, db, learning_rate);
  sub(res, b, res);
  b = res;
}

double LinearLayer::randGaussian()
{
  double r = ((double)rand() / (RAND_MAX));
  return ((double)rand() / (RAND_MAX)) > 0.5 ? sqrt(-2 * log(r)) : -1 * sqrt(-2 * log(r));
}