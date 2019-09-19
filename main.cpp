#include "utils/TimeElapsed.h"
#include "utils/MatrixFunctions.h"
#include "MLP/MLP.h"
#include "CNN/CNN.h"
#include "utils/CSVReader.h"
#include "common.h"

void conv(vector<vector<double>> &result, vector<vector<double>> data, vector<vector<double>> filter, int stride);
void pool(vector<vector<double>> &result, int size, int stride);
void ReLU(vector<vector<double>> &result);

int main()
{
  startTimer();
  cout << fixed << setprecision(10);

  int filters = 3;
  int classes = 2; // Output types

  vector<vector<double>> image1;
  image1.push_back({-1, -1, -1, -1, -1, -1, -1, -1, -1});
  image1.push_back({-1, 1, -1, -1, -1, -1, -1, 1, -1});
  image1.push_back({-1, -1, 1, -1, -1, -1, 1, -1, -1});
  image1.push_back({-1, -1, -1, 1, -1, 1, -1, -1, -1});
  image1.push_back({-1, -1, -1, -1, 1, -1, -1, -1, -1});
  image1.push_back({-1, -1, -1, 1, -1, 1, -1, -1, -1});
  image1.push_back({-1, -1, 1, -1, -1, -1, 1, -1, -1});
  image1.push_back({-1, 1, -1, -1, -1, -1, -1, 1, -1});
  image1.push_back({-1, -1, -1, -1, -1, -1, -1, -1, -1});
  vector<vector<double>> filter1;
  filter1.push_back({1, -1, -1});
  filter1.push_back({-1, 1, -1});
  filter1.push_back({-1, -1, 1});
  vector<vector<double>> filter2;
  filter2.push_back({1, -1, 1});
  filter2.push_back({-1, 1, -1});
  filter2.push_back({1, -1, 1});
  vector<vector<double>> filter3;
  filter3.push_back({-1, -1, 1});
  filter3.push_back({-1, 1, -1});
  filter3.push_back({1, -1, -1});

  vector<vector<double>> res1;
  vector<vector<double>> res2;
  vector<vector<double>> res3;

  conv(res1, image1, filter1, 1);
  conv(res2, image1, filter2, 1);
  conv(res3, image1, filter3, 1);

  ReLU(res1);
  ReLU(res2);
  ReLU(res3);

  pool(res1, 2, 2);
  pool(res2, 2, 2);
  pool(res3, 2, 2);

  pool(res1, 2, 2);
  pool(res2, 2, 2);
  pool(res3, 2, 2);

  // for (auto &row : res1)
  // {
  //   for (auto &col : row)
  //   {
  //     cout << col << " ";
  //   }
  //   cout << endl;
  // }
  // cout << endl;
  // for (auto &row : res2)
  // {
  //   for (auto &col : row)
  //   {
  //     cout << col << " ";
  //   }
  //   cout << endl;
  // }
  // cout << endl;
  // for (auto &row : res3)
  // {
  //   for (auto &col : row)
  //   {
  //     cout << col << " ";
  //   }
  //   cout << endl;
  // }
  // cout << endl;

  vector<vector<double>> X_train;
  // X_train.push_back({0, 0, 1, 1});
  // X_train.push_back({0, 1, 0, 1});
  vector<vector<double>> Y_train;
  // Y_train.push_back({1, 0, 0, 1});
  // Y_train.push_back({0, 0, 1, 1});

  vector<double> temp1;
  vector<vector<double>> total;
  for (int j = 0; j < res1.size(); j++)
  {
    for (int k = 0; k < res1[0].size(); k++)
    {
      temp1.push_back(res1[j][k]);
    }
  }
  for (int j = 0; j < res2.size(); j++)
  {
    for (int k = 0; k < res2[0].size(); k++)
    {
      temp1.push_back(res2[j][k]);
    }
  }
  for (int j = 0; j < res3.size(); j++)
  {
    for (int k = 0; k < res3[0].size(); k++)
    {
      temp1.push_back(res3[j][k]);
    }
  }
  total.push_back(temp1);
  // for (auto &row : total)
  // {
  //   for (auto &col : row)
  //   {
  //     cout << col << " ";
  //   }
  //   cout << endl;
  // }
  // cout << endl;

  transpose(X_train, total);
  Y_train.push_back({1});
  Y_train.push_back({0});

  MLP model{1};
  model.addDenseLayer(ceil(X_train.size() / 1.5), X_train.size(), "xavier");
  model.addDenseLayer(classes, 0, "xavier");

  for (int i = 0; i < 100; i++)
  {
    model.train(100, X_train, Y_train);
    // cout << "Cost at epoch " << model.getEpoch() << ": " << model.getCost() << endl;
  }
  cout << "Final cost at epoch " << model.getEpoch() << ": " << model.getCost() << endl;
  vector<vector<double>> pred;
  vector<vector<double>> Xs;
  Xs.push_back({1});
  Xs.push_back({0.55});
  Xs.push_back({.55});
  Xs.push_back({1});
  Xs.push_back({1});
  Xs.push_back({0.55});
  Xs.push_back({0.55});
  Xs.push_back({0.55});
  Xs.push_back({0.55});
  Xs.push_back({1});
  Xs.push_back({1});
  Xs.push_back({0.55});
  model.predict(pred, Xs);
  for (auto &row : pred)
  {
    for (auto &col : row)
    {
      cout << col << " ";
    }
    cout << endl;
  }
  cout << endl;
  vector<vector<double>> test;
  model.softmax(test, pred);
  for (auto &row : test)
  {
    for (auto &col : row)
    {
      cout << col << " ";
    }
    cout << endl;
  }

  vector<string> test2;
  getRow(test2, "mnist_train", 0);
  // cout << stoi(test2[0]) << endl;
  int label = stoi(test2[0]);
  vector<vector<double>> mnist1;
  int index = 1;
  for (int i = 0; i < 28; i++)
  {
    vector<double> temp;
    for (int j = 0; j < 28; j++)
    {
      temp.push_back(stoi(test2[index]));
      index++;
    }
    mnist1.push_back(temp);
  }
  cout << fixed << setprecision(8);
  double mean1;
  meanAll(mean1, mnist1);
  int realMean = (int)mean1;
  vector<vector<double>> resulty(28, vector<double>(28, 0));
  for (int i = 0; i < 28; i++)
  {
    for (int j = 0; j < 28; j++)
    {
      resulty[i][j] = mnist1[i][j] - realMean;
    }
  }
  double stdDDD;
  stdAll(stdDDD, resulty);
  int realSTD = (int)stdDDD;
  for (int i = 0; i < 28; i++)
  {
    for (int j = 0; j < 28; j++)
    {
      resulty[i][j] = resulty[i][j] / realSTD;
    }
  }
  vector<vector<double>> lab(10, vector<double>(1, 0));
  lab[label][0] = 1;

  vector<vector<vector<double>>> data;
  data.push_back(resulty);
  // cout << fixed << setprecision(2);
  cout << endl;
  CNN first;
  first.conv(data, lab);

  finish();
  // system("pause");
}

void conv(vector<vector<double>> &result, vector<vector<double>> data, vector<vector<double>> filter, int stride)
{
  int filterRow = filter.size();
  int filterCol = filter[0].size();
  int padding = data[0].size() % stride;
  // if (stride == 1)
  // {
  //   padding = (filterCol - 1) / 2;
  // }
  // else
  // {
  //   int padding = ceil(((stride - 1) * data[0].size() - stride + filterCol) / 2.0);
  // }
  if (padding > 0)
  {
    int oriSize = data.size();
    for (int i = 0; i < padding; i++)
    {
      for (int j = 0; j < oriSize; j++)
      {
        // data[j].insert(data[j].begin(), 0);
        data[j].push_back(0);
      }
      // data.insert(data.begin(), vector<double>(oriSize + 1 + padding, 0));
      // data.push_back(vector<double>(oriSize + 1 + padding, 0));
      data.push_back(vector<double>(oriSize + padding, 0));
    }
  }
  int resultW = (data[0].size() - filterCol) / stride + 1;
  int resultH = (data.size() - filterRow) / stride + 1;
  vector<vector<double>> res(resultH, vector<double>(resultW, 0));
  for (int i = 0; i < resultH; i++)
  {
    for (int j = 0; j < resultW; j++)
    {
      double sum = 0;
      for (int k = 0; k < filterCol; k++)
      {
        for (int l = 0; l < filterRow; l++)
        {
          sum += filter[l][k] * data[l + i * stride][k + j * stride];
        }
      }
      res[i][j] = sum / (filterRow * filterCol);
    }
  }
  result = res;
}

void pool(vector<vector<double>> &result, int size, int stride)
{
  if (result.size() == 0)
  {
    cout << "ERROR: Input must be initialized" << endl;
  }
  int filterRow = size;
  int filterCol = size;
  int padding = result[0].size() % size;
  // if (stride == 1)
  // {
  // padding = (filterCol - 1) / 2;
  // }
  // else
  // {
  //   int padding = ceil(((stride - 1) * result[0].size() - stride + filterCol) / 2.0);
  // }
  // cout << padding << endl;
  if (padding > 0)
  {
    int oriSize = result.size();
    for (int i = 0; i < padding; i++)
    {
      for (int j = 0; j < oriSize; j++)
      {
        // result[j].insert(result[j].begin(), 0);
        result[j].push_back(0);
      }
      // result.insert(result.begin(), vector<double>(oriSize + 1 + padding, 0));
      // result.push_back(vector<double>(oriSize + 1 + padding, 0));
      result.push_back(vector<double>(oriSize + padding, 0));
    }
  }
  int resultW = (result[0].size() - filterCol) / stride + 1;
  int resultH = (result.size() - filterRow) / stride + 1;
  vector<vector<double>> res(resultH, vector<double>(resultW, 0));
  for (int i = 0; i < res.size(); i++)
  {
    for (int j = 0; j < res[0].size(); j++)
    {
      double max = RAND_MAX * -1;
      for (int k = 0; k < filterCol; k++)
      {
        for (int l = 0; l < filterRow; l++)
        {
          max = result[l + i * stride][k + j * stride] > max ? result[l + i * stride][k + j * stride] : max;
        }
      }
      res[i][j] = max;
    }
  }
  result = res;
}

void ReLU(vector<vector<double>> &result)
{
  for (int i = 0; i < result.size(); i++)
  {
    for (int j = 0; j < result[0].size(); j++)
    {
      if (result[i][j] < 0)
      {
        result[i][j] = 0;
      }
    }
  }
}