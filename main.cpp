#include "utils/TimeElapsed.h"
#include "utils/MatrixFunctions.h"
#include "CNN/CNN.h"
#include "utils/CSVReader.h"
#include "common.h"

int main()
{
  startTimer();
  cout << fixed << setprecision(10);

  vector<string> test2;
  getRow(test2, "mnist_train", 0);
  // cout << stoi(test2[0]) << endl;
  vector<int> labels;
  labels.push_back(stoi(test2[0]));
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
  vector<vector<vector<vector<double>>>> data;
  data.push_back({resulty});

  vector<vector<int>> params;
  params.push_back({8, 1, 5, 5}); // f1
  params.push_back({8, 8, 5, 5}); // f2
  params.push_back({128, 800}); // w3
  params.push_back({10, 128}); // w4
  params.push_back({8, 1}); // b1
  params.push_back({8, 1}); // b2
  params.push_back({128, 1}); // b3
  params.push_back({10, 1}); // b4
  CNN first{params};
  first.train(1, data.size(), data, labels);
  finish();
  // system("pause");
}
