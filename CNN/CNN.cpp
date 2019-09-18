#include "CNN.h"
#include "../utils/MatrixFunctions.h"
#include "Forward.h"
#include "Backward.h"
#include "../common.h"

void CNN::test(vector<vector<vector<double>>> image)
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
}