#include "Activation.h"
#include "../utils/MatrixFunctions.h"
#include "../common.h"

void Activation::activationForward(vector<vector<double>> Z)
{
  int row = Z.size();
  int col = Z[0].size();
  vector<vector<double>> res(row, vector<double>(col, 0));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      res[i][j] = sigmoid(Z[i][j]);
    }
  }
  A = res;
}
void Activation::activationBackward(vector<vector<double>> upstream_grad)
{
  int row = A.size();
  int col = A[0].size();
  vector<vector<double>> res(row, vector<double>(col, 0));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      res[i][j] = dSigmoid(A[i][j]);
    }
  }
  multMatrices(res, upstream_grad, res);
  dZ = res;
}