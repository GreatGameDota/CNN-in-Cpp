#include "utils/TimeElapsed.h"
#include "utils/MatrixFunctions.h"
#include "CNN/CNN.h"
#include "common.h"

int main()
{
  // startTimer();
  CNN model;
  int dataAmount = 1;
  cout << "Training model with " << dataAmount << " images...\n\n";
  model.train(2, dataAmount);
  model.exportData("output.txt");
  // finish();
  system("pause");
}
