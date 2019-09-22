#include "utils/TimeElapsed.h"
#include "utils/MatrixFunctions.h"
#include "CNN/CNN.h"
#include "common.h"

int main()
{
  // startTimer();
  CNN model;
  int dataAmount = 1;
  // model.importData("output.txt");
  model.train(2, dataAmount);
  model.exportData("output.txt");
  // finish();
  system("pause");
}
