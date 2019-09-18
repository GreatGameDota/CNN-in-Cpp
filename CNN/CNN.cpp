#include "CNN.h"
#include "../utils/MatrixFunctions.h"
#include "Forward.h"
#include "Backward.h"
#include "../common.h"

void CNN::conv(vector<vector<vector<double>>> image, int label)
{
  vector<vector<vector<double>>> res;
  // vector<vector<vector<double>>> image(1, vector<vector<double>>(28, vector<double>(28, 0)));
  vector<vector<vector<vector<double>>>> filter(8, vector<vector<vector<double>>>(1, vector<vector<double>>(5, vector<double>(5, 1))));
  vector<vector<double>> bias(8, vector<double>(1, 0));
  convolution(res, image, filter, bias);
  ReLU(res);
  vector<vector<vector<vector<double>>>> filter2(8, vector<vector<vector<double>>>(8, vector<vector<double>>(5, vector<double>(5, 1))));
  convolution(res, res, filter2, bias);
  ReLU(res);
  maxpool(res, res);
  // Fully connected
  vector<vector<double>> flat;
  for (auto &row : res)
  {
    for (auto &col : row)
    {
      for (auto &ele : col)
      {
        flat.push_back({ele});
      }
    }
  }
  vector<vector<double>> w3(128, vector<double>(800, .00001));
  vector<vector<double>> b3(128, vector<double>(1, 0));
  vector<vector<double>> z;
  dot(z, w3, flat);
  add(z, z, b3);
  for (int i = 0; i < z.size(); i++)
  {
    for (int j = 0; j < z[0].size(); j++)
    {
      if (z[i][j] < 0)
        z[i][j] = 0;
    }
  }
  vector<vector<double>> w4(10, vector<double>(128, .00001));
  vector<vector<double>> b4(10, vector<double>(1, 0));
  vector<vector<double>> out;
  dot(out, w4, z);
  add(out, out, b4);

  vector<vector<double>> probs;
  softmax(probs, out);
  double loss;
  categoricalCrossEntropy(loss, probs, label);
  cout << loss << endl;
}

double CNN::randGaussian()
{
  double r = ((double)rand() / (RAND_MAX));
  return ((double)rand() / (RAND_MAX)) > 0.5 ? sqrt(-2 * log(r)) : -1 * sqrt(-2 * log(r));
}