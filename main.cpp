#include "utils/TimeElapsed.h"
#include "utils/MatrixFunctions.h"
#include "CNN/CNN.h"
#include "common.h"

int main()
{
  CNN model;
  int dataAmount = 100;
  // model.importData("output.txt");
  model.train(2, dataAmount);
  model.exportData("output.txt");

  cout << "Test 1:" << endl;
  vector<vector<vector<double>>> image;
  int label;
  model.getMNISTData(image, label, 0, "csv/mnist_test.csv");
  cout << "Correct output is: " << label << endl;
  vector<vector<double>> probs;
  model.predict(probs, image);
  cout << "Output: " << endl;
  int num = 0;
  for (auto &row : probs)
  {
    for (auto &col : row)
    {
      cout << num << ": " << col << endl;
      num++;
    }
  }

  system("pause");
}
