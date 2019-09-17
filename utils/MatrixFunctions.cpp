#include "../common.h"
#include "MatrixFunctions.h"

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
  vector<vector<double>> res(rowA, vector<double>(colB, 0));
  for (int i = 0; i < rowA; i++)
  {
    for (int j = 0; j < colB; j++)
    {
      double sum = 0;
      for (int k = 0; k < colA; k++)
      {
        sum += matrixA[i][k] * matrixB[k][j];
      }
      res[i][j] = sum;
    }
  }
  result = res;
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