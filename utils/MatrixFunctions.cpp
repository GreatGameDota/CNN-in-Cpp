#include "../common.h"
#include "MatrixFunctions.h"
#include <numeric>

void dot(vector<vector<double>> &result, vector<vector<double>> matrixA, vector<vector<double>> matrixB)
{
  int rowA = matrixA.size();
  int colA = matrixA[0].size();
  int rowB = matrixB.size();
  int colB = matrixB[0].size();
  if (colA != rowB)
  {
    cout << "Dot Function error: Dimension Mismatch" << endl;
  }
  result = vector<vector<double>>(rowA, vector<double>(colB, 0));
  // vector<vector<double>> res(rowA, vector<double>(colB, 0));
  for (int i = 0; i < rowA; i++)
  {
    for (int j = 0; j < colB; j++)
    {
      double sum = 0;
      for (int k = 0; k < colA; k++)
      {
        sum += matrixA[i][k] * matrixB[k][j];
      }
      result[i][j] = sum;
    }
  }
  // result = res;
}

void transpose(vector<vector<double>> &result, vector<vector<double>> matrix)
{
  int row = matrix.size();
  int col = matrix[0].size();
  vector<vector<double>> res(col, vector<double>(row, 0));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      res[j][i] = matrix[i][j];
    }
  }
  result = res;
}

void sub(vector<vector<double>> &result, vector<vector<double>> matrixA, vector<vector<double>> matrixB)
{
  int rowA = matrixA.size();
  int colA = matrixA[0].size();
  int rowB = matrixB.size();
  int colB = matrixB[0].size();
  if (rowA * colA != rowB * colB)
  {
    cout << "Sub Function error: Dimensions are not the same" << endl;
  }
  vector<vector<double>> res(rowA, vector<double>(colA, 0));
  for (int i = 0; i < rowA; i++)
  {
    for (int j = 0; j < colA; j++)
    {
      res[i][j] = matrixA[i][j] - matrixB[i][j];
    }
  }
  result = res;
}

void mult(vector<vector<double>> &result, vector<vector<double>> matrix, double n)
{
  int row = matrix.size();
  int col = matrix[0].size();
  vector<vector<double>> res(row, vector<double>(col, 0));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      res[i][j] = matrix[i][j] * n;
    }
  }
  result = res;
}

void multMatrices(vector<vector<double>> &result, vector<vector<double>> matrixA, vector<vector<double>> matrixB)
{
  int rowA = matrixA.size();
  int colA = matrixA[0].size();
  int rowB = matrixB.size();
  int colB = matrixB[0].size();
  if (rowA * colA != rowB * colB)
  {
    cout << "MultMatrices Function error: Dimensions are not the same" << endl;
  }
  vector<vector<double>> res(rowA, vector<double>(colA, 0));
  for (int i = 0; i < rowA; i++)
  {
    for (int j = 0; j < colA; j++)
    {
      res[i][j] = matrixA[i][j] * matrixB[i][j];
    }
  }
  result = res;
}

void multMatrices3D(vector<vector<vector<double>>> &result, vector<vector<vector<double>>> matrixA, vector<vector<vector<double>>> matrixB)
{
  int rowA = matrixA.size();
  int colA = matrixA[0].size();
  int rowB = matrixB.size();
  int colB = matrixB[0].size();
  if (rowA * colA * matrixA[0][0].size() != rowB * colB * matrixB[0][0].size())
  {
    cout << "MultMatrices3D Function error: Dimensions are not the same" << endl;
  }
  result = vector<vector<vector<double>>>(rowA, vector<vector<double>>(colA, vector<double>(matrixA[0][0].size(), 0)));
  // vector<vector<vector<double>>> res(rowA, vector<vector<double>>(colA, vector<double>(matrixA[0][0].size(), 0)));
  for (int i = 0; i < rowA; i++)
  {
    for (int j = 0; j < colA; j++)
    {
      for (int k = 0; k < matrixA[0][0].size(); k++)
        result[i][j][k] = matrixA[i][j][k] * matrixB[i][j][k];
    }
  }
  // result = res;
}

void add(vector<vector<double>> &result, vector<vector<double>> matrixA, vector<vector<double>> matrixB)
{
  int rowA = matrixA.size();
  int colA = matrixA[0].size();
  int rowB = matrixB.size();
  int colB = matrixB[0].size();
  int matrixALength = rowA + colA;
  int matrixBLength = rowB + colB;
  // Check if dimensions need broadcasting
  int tempColB = colB;
  if (colB == 1)
  {
    colB = colA;
  }
  int tempColA = colA;
  if (colA == 1)
  {
    colA = colB;
  }
  if (colA != colB)
  {
    cout << "Add Function error: Dimension Mismatch" << endl;
  }
  colA = tempColA;
  colB = tempColB;
  // Broadcasting
  vector<vector<double>> mA = matrixA;
  vector<vector<double>> mB = matrixB;
  int resultRow;
  int resultCol;
  if (rowB < rowA)
  {
    for (int i = 0; i < rowA - rowB; i++)
    {
      mB.push_back(mB[0]);
    }
  }
  else if (rowA < rowB)
  {
    for (int i = 0; i < rowB - rowA; i++)
    {
      mA.push_back(mA[0]);
    }
  }
  if (colB < colA)
  {
    for (int i = 0; i < colA - colB; i++)
    {
      for (int j = 0; j < rowB; j++)
      {
        mB[j].push_back(mB[j][0]);
      }
    }
  }
  else if (colA < colB)
  {
    for (int i = 0; i < colB - colA; i++)
    {
      for (int j = 0; j < rowA; j++)
      {
        mA[j].push_back(mA[j][0]);
      }
    }
  }
  resultRow = max(rowA, rowB);
  resultCol = max(colA, colB);
  vector<vector<double>> res(resultRow, vector<double>(resultCol, 0));
  for (int i = 0; i < resultRow; i++)
  {
    for (int j = 0; j < resultCol; j++)
    {
      res[i][j] = mA[i][j] + mB[i][j];
    }
  }
  result = res;
}

void sum(vector<vector<double>> &result, vector<vector<double>> matrix, int axis)
{
  // Axis meaning whether to sum x or y (row or col) (2 for y, 1 for x, 0 for everything)
  vector<vector<double>> res;
  int row = matrix.size();
  int col = matrix[0].size();
  if (axis == 1)
  {
    res = vector<vector<double>>(row, vector<double>(1, 0));
    for (int i = 0; i < row; i++)
    {
      double sum = 0;
      for (int j = 0; j < col; j++)
      {
        sum += matrix[i][j];
      }
      res[i][0] = sum;
    }
  }
  else if (axis == 2)
  {
    res = vector<vector<double>>(1, vector<double>(col, 0));
    for (int i = 0; i < col; i++)
    {
      double sum = 0;
      for (int j = 0; j < row; j++)
      {
        sum += matrix[j][i];
      }
      res[0][i] = sum;
    }
  }
  else
  {
    res = vector<vector<double>>(1, vector<double>(1, 0));
    double sum = 0;
    for (int i = 0; i < col; i++)
    {
      for (int j = 0; j < row; j++)
      {
        sum += matrix[j][i];
      }
    }
    res[0][0] = sum;
  }
  result = res;
}

void sum3D(vector<vector<vector<double>>> &result, vector<vector<vector<double>>> matrix, int axis)
{
  int row = matrix[0].size();
  int col = matrix[0][0].size();
  result = vector<vector<vector<double>>>(1, vector<vector<double>>(1, vector<double>(1, 0)));
  // vector<vector<vector<double>>> res(1, vector<vector<double>>(1, vector<double>(1, 0)));
  double sum = 0;
  for (int i = 0; i < matrix.size(); i++)
  {
    for (int j = 0; j < row; j++)
    {
      for (int k = 0; k < col; k++)
        sum += matrix[i][j][k];
    }
  }
  result[0][0][0] = sum;
  // result = res;
}

void square(vector<vector<double>> &result, vector<vector<double>> matrix)
{
  int row = matrix.size();
  int col = matrix[0].size();
  vector<vector<double>> res(row, vector<double>(col, 0));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      res[i][j] = matrix[i][j] * matrix[i][j];
    }
  }
  result = res;
}

void meanAll(double &result, vector<vector<double>> matrix)
{
  int row = matrix.size();
  int col = matrix[0].size();
  double sum = 0;
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      sum += matrix[i][j];
    }
  }
  result = sum / (row * col);
}

void stdAll(double &result, vector<vector<double>> matrix)
{
  int row = matrix.size();
  int col = matrix[0].size();
  double firstMean;
  meanAll(firstMean, matrix);
  vector<vector<double>> res(row, vector<double>(col, 0));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      res[i][j] = matrix[i][j] - firstMean;
    }
  }
  square(res, res);
  double secondMean;
  meanAll(secondMean, res);
  result = sqrt(secondMean);
}

void mult3D(vector<vector<vector<double>>> &result, vector<vector<vector<double>>> matrix, double n)
{
  int row = matrix.size();
  int col = matrix[0].size();
  vector<vector<vector<double>>> res(row, vector<vector<double>>(col, vector<double>(matrix[0][0].size(), 0)));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      for (int k = 0; k < matrix[0][0].size(); k++)
      {
        res[i][j][k] = matrix[i][j][k] * n;
      }
    }
  }
  result = res;
}

void add3D(vector<vector<vector<double>>> &result, vector<vector<vector<double>>> matrixA, vector<vector<vector<double>>> matrixB)
{
  int row = matrixA.size();
  int col = matrixA[0].size();
  if (row * col * matrixA[0][0].size() != matrixB.size() * matrixB[0].size() * matrixB[0][0].size())
  {
    cout << "Add3D Function error: Dimensions are not the same" << endl;
  }
  vector<vector<vector<double>>> res(row, vector<vector<double>>(col, vector<double>(matrixA[0][0].size(), 0)));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      for (int k = 0; k < matrixA[0][0].size(); k++)
      {
        res[i][j][k] = matrixA[i][j][k] + matrixB[i][j][k];
      }
    }
  }
  result = res;
}

void add4D(vector<vector<vector<vector<double>>>> &result, vector<vector<vector<vector<double>>>> matrixA, vector<vector<vector<vector<double>>>> matrixB)
{
  int row = matrixA.size();
  int col = matrixA[0].size();
  if (row * col * matrixA[0][0].size() * matrixA[0][0][0].size() != matrixB.size() * matrixB[0].size() * matrixB[0][0].size() * matrixB[0][0][0].size())
  {
    cout << "Add4D Function error: Dimensions are not the same" << endl;
  }
  vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(matrixA[0][0].size(), vector<double>(matrixA[0][0][0].size(), 0))));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      for (int k = 0; k < matrixA[0][0].size(); k++)
      {
        for (int l = 0; l < matrixA[0][0][0].size(); l++)
        {
          res[i][j][k][l] = matrixA[i][j][k][l] + matrixB[i][j][k][l];
        }
      }
    }
  }
  result = res;
}

void mult4D(vector<vector<vector<vector<double>>>> &result, vector<vector<vector<vector<double>>>> matrix, double n)
{
  int row = matrix.size();
  int col = matrix[0].size();
  vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(matrix[0][0].size(), vector<double>(matrix[0][0][0].size(), 0))));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      for (int k = 0; k < matrix[0][0].size(); k++)
      {
        for (int l = 0; l < matrix[0][0][0].size(); l++)
        {
          res[i][j][k][l] = matrix[i][j][k][l] * n;
        }
      }
    }
  }
  result = res;
}

void divi4D(vector<vector<vector<vector<double>>>> &result, vector<vector<vector<vector<double>>>> matrix, double n)
{
  int row = matrix.size();
  int col = matrix[0].size();
  vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(matrix[0][0].size(), vector<double>(matrix[0][0][0].size(), 0))));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      for (int k = 0; k < matrix[0][0].size(); k++)
      {
        for (int l = 0; l < matrix[0][0][0].size(); l++)
        {
          res[i][j][k][l] = matrix[i][j][k][l] / n;
        }
      }
    }
  }
  result = res;
}

void square4D(vector<vector<vector<vector<double>>>> &result, vector<vector<vector<vector<double>>>> matrix)
{
  int row = matrix.size();
  int col = matrix[0].size();
  vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(matrix[0][0].size(), vector<double>(matrix[0][0][0].size(), 0))));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      for (int k = 0; k < matrix[0][0].size(); k++)
      {
        for (int l = 0; l < matrix[0][0][0].size(); l++)
        {
          res[i][j][k][l] = matrix[i][j][k][l] * matrix[i][j][k][l];
        }
      }
    }
  }
  result = res;
}

void addN4D(vector<vector<vector<vector<double>>>> &result, vector<vector<vector<vector<double>>>> matrix, double n)
{
  int row = matrix.size();
  int col = matrix[0].size();
  vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(matrix[0][0].size(), vector<double>(matrix[0][0][0].size(), 0))));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      for (int k = 0; k < matrix[0][0].size(); k++)
      {
        for (int l = 0; l < matrix[0][0][0].size(); l++)
        {
          res[i][j][k][l] = matrix[i][j][k][l] + n;
        }
      }
    }
  }
  result = res;
}

void divi4DMat(vector<vector<vector<vector<double>>>> &result, vector<vector<vector<vector<double>>>> matrixA, vector<vector<vector<vector<double>>>> matrixB)
{
  int row = matrixA.size();
  int col = matrixA[0].size();
  if (row * col * matrixA[0][0].size() * matrixA[0][0][0].size() != matrixB.size() * matrixB[0].size() * matrixB[0][0].size() * matrixB[0][0][0].size())
  {
    cout << "Divi4DMat Function error: Dimensions are not the same" << endl;
  }
  vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(matrixA[0][0].size(), vector<double>(matrixA[0][0][0].size(), 0))));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      for (int k = 0; k < matrixA[0][0].size(); k++)
      {
        for (int l = 0; l < matrixA[0][0][0].size(); l++)
        {
          res[i][j][k][l] = matrixA[i][j][k][l] / matrixB[i][j][k][l];
        }
      }
    }
  }
  result = res;
}

void sub4DMat(vector<vector<vector<vector<double>>>> &result, vector<vector<vector<vector<double>>>> matrixA, vector<vector<vector<vector<double>>>> matrixB)
{
  int row = matrixA.size();
  int col = matrixA[0].size();
  if (row * col * matrixA[0][0].size() * matrixA[0][0][0].size() != matrixB.size() * matrixB[0].size() * matrixB[0][0].size() * matrixB[0][0][0].size())
  {
    cout << "Sub4DMat Function error: Dimensions are not the same" << endl;
  }
  vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(matrixA[0][0].size(), vector<double>(matrixA[0][0][0].size(), 0))));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      for (int k = 0; k < matrixA[0][0].size(); k++)
      {
        for (int l = 0; l < matrixA[0][0][0].size(); l++)
        {
          res[i][j][k][l] = matrixA[i][j][k][l] - matrixB[i][j][k][l];
        }
      }
    }
  }
  result = res;
}

void sqrt4D(vector<vector<vector<vector<double>>>> &result, vector<vector<vector<vector<double>>>> matrix)
{
  int row = matrix.size();
  int col = matrix[0].size();
  vector<vector<vector<vector<double>>>> res(row, vector<vector<vector<double>>>(col, vector<vector<double>>(matrix[0][0].size(), vector<double>(matrix[0][0][0].size(), 0))));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      for (int k = 0; k < matrix[0][0].size(); k++)
      {
        for (int l = 0; l < matrix[0][0][0].size(); l++)
        {
          res[i][j][k][l] = sqrt(matrix[i][j][k][l]);
        }
      }
    }
  }
  result = res;
}

void divi2D(vector<vector<double>> &result, vector<vector<double>> matrix, double n)
{
  int row = matrix.size();
  int col = matrix[0].size();
  vector<vector<double>> res(row, vector<double>(col, 0));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      res[i][j] = matrix[i][j] / n;
    }
  }
  result = res;
}

void addN2D(vector<vector<double>> &result, vector<vector<double>> matrix, double n)
{
  int row = matrix.size();
  int col = matrix[0].size();
  vector<vector<double>> res(row, vector<double>(col, 0));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      res[i][j] = matrix[i][j] + n;
    }
  }
  result = res;
}

void divi2DMat(vector<vector<double>> &result, vector<vector<double>> matrixA, vector<vector<double>> matrixB)
{
  int row = matrixA.size();
  int col = matrixA[0].size();
  if (row * col != matrixB.size() * matrixB[0].size())
  {
    cout << "Divi2DMat Function error: Dimensions are not the same" << endl;
  }
  vector<vector<double>> res(row, vector<double>(col, 0));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      res[i][j] = matrixA[i][j] / matrixB[i][j];
    }
  }
  result = res;
}

void sub2DMat(vector<vector<double>> &result, vector<vector<double>> matrixA, vector<vector<double>> matrixB)
{
  int row = matrixA.size();
  int col = matrixA[0].size();
  if (row * col != matrixB.size() * matrixB[0].size())
  {
    cout << "Sub2DMat Function error: Dimensions are not the same" << endl;
  }
  vector<vector<double>> res(row, vector<double>(col, 0));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      res[i][j] = matrixA[i][j] - matrixB[i][j];
    }
  }
  result = res;
}

void sqrt2D(vector<vector<double>> &result, vector<vector<double>> matrix)
{
  int row = matrix.size();
  int col = matrix[0].size();
  vector<vector<double>> res(row, vector<double>(col, 0));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      res[i][j] = sqrt(matrix[i][j]);
    }
  }
  result = res;
}
