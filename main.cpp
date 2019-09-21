#include "utils/TimeElapsed.h"
#include "utils/MatrixFunctions.h"
#include "CNN/CNN.h"
#include "common.h"

int main()
{
  // startTimer();
  CNN model;
  int dataAmount = 50;
  cout << "Training model with " << dataAmount << " images...\n\n";
  model.train(2, dataAmount);
  // finish();
  system("pause");
}
