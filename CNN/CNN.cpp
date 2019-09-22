#include "CNN.h"
#include "../utils/MatrixFunctions.h"
#include "../utils/CSVReader.h"
#include "../utils/TimeElapsed.h"
#include "Forward.h"
#include "Backward.h"
#include "../common.h"

void CNN::train(int epochs, int dataAmount)
{
  int imageAmount = dataAmount;
  int epoch = 0;
  int batchSize = 32;
  vector<double> costs;
  cout << "Training model with " << imageAmount << " images...\n\n";
  for (int i = 0; i < epochs; i++)
  {
    cout << "\rEpoch: 0 Cost: N/A | ";
    cout << "0% | 0/" << (imageAmount / batchSize + 1) << " | ";
    cout << "[--m --s --ms per iter]";
    for (int j = 0; j < imageAmount / batchSize + 1; j++)
    {
      epoch++;
      startTimer();
      adamGD(imageAmount, costs);
      cout << fixed << setprecision(10);
      cout << "\rEpoch: " << epoch << " Cost: " << costs.back() << " | ";
      cout << (int)(((double)(j + 1) / (imageAmount / batchSize + 1)) * 100) << "% | " << j + 1 << "/" << (imageAmount / batchSize + 1) << " | ";
      long min, sec, milli;
      finish(min, sec, milli);
      cout << "[" << min << "m " << sec << "s " << milli << "ms per iter]";
    }
    cout << endl;
  }
}

void CNN::adamGD(int imageAmount, vector<double> &cost)
{
  double _cost = 0;
  // Initialize gradients and momentum, RMS params
  vector<vector<vector<vector<double>>>> df1(params[0][0], vector<vector<vector<double>>>(params[0][1], vector<vector<double>>(params[0][2], vector<double>(params[0][3], 0))));
  vector<vector<vector<vector<double>>>> df2(params[1][0], vector<vector<vector<double>>>(params[1][1], vector<vector<double>>(params[1][2], vector<double>(params[1][3], 0))));
  vector<vector<double>> dw3(params[2][0], vector<double>(params[2][1], 0));
  vector<vector<double>> dw4(params[3][0], vector<double>(params[3][1], 0));
  vector<vector<double>> db1(params[4][0], vector<double>(params[4][1], 0));
  vector<vector<double>> db2(params[5][0], vector<double>(params[5][1], 0));
  vector<vector<double>> db3(params[6][0], vector<double>(params[6][1], 0));
  vector<vector<double>> db4(params[7][0], vector<double>(params[7][1], 0));

  vector<vector<vector<vector<double>>>> v1(params[0][0], vector<vector<vector<double>>>(params[0][1], vector<vector<double>>(params[0][2], vector<double>(params[0][3], 0))));
  vector<vector<vector<vector<double>>>> v2(params[1][0], vector<vector<vector<double>>>(params[1][1], vector<vector<double>>(params[1][2], vector<double>(params[1][3], 0))));
  vector<vector<double>> v3(params[2][0], vector<double>(params[2][1], 0));
  vector<vector<double>> v4(params[3][0], vector<double>(params[3][1], 0));
  vector<vector<double>> bv1(params[4][0], vector<double>(params[4][1], 0));
  vector<vector<double>> bv2(params[5][0], vector<double>(params[5][1], 0));
  vector<vector<double>> bv3(params[6][0], vector<double>(params[6][1], 0));
  vector<vector<double>> bv4(params[7][0], vector<double>(params[7][1], 0));

  vector<vector<vector<vector<double>>>> s1(params[0][0], vector<vector<vector<double>>>(params[0][1], vector<vector<double>>(params[0][2], vector<double>(params[0][3], 0))));
  vector<vector<vector<vector<double>>>> s2(params[1][0], vector<vector<vector<double>>>(params[1][1], vector<vector<double>>(params[1][2], vector<double>(params[1][3], 0))));
  vector<vector<double>> s3(params[2][0], vector<double>(params[2][1], 0));
  vector<vector<double>> s4(params[3][0], vector<double>(params[3][1], 0));
  vector<vector<double>> bs1(params[4][0], vector<double>(params[4][1], 0));
  vector<vector<double>> bs2(params[5][0], vector<double>(params[5][1], 0));
  vector<vector<double>> bs3(params[6][0], vector<double>(params[6][1], 0));
  vector<vector<double>> bs4(params[7][0], vector<double>(params[7][1], 0));

  for (int i = 0; i < imageAmount; i++)
  {
    vector<vector<vector<double>>> image;
    int _label;
    getMNISTData(image, _label, i, "csv/mnist_train.csv");
    vector<vector<double>> label(10, vector<double>(1, 0));
    label[_label][0] = 1;
    double loss;
    vector<vector<vector<vector<double>>>> _df1;
    vector<vector<vector<vector<double>>>> _df2;
    vector<vector<double>> _dw3;
    vector<vector<double>> _dw4;
    vector<vector<double>> _db1;
    vector<vector<double>> _db2;
    vector<vector<double>> _db3;
    vector<vector<double>> _db4;
    conv(loss, _df1, _df2, _dw3, _dw4, _db1, _db2, _db3, _db4, image, label);
    _cost += loss;
    add4D(df1, df1, _df1);
    add4D(df2, df2, _df2);
    add(dw3, dw3, _dw3);
    add(dw4, dw4, _dw4);
    add(db1, db1, _db1);
    add(db2, db2, _db2);
    add(db3, db3, _db3);
    add(db4, db4, _db4);
  }

  // Parameter update (Adam Gradient Descent)
  // f1
  vector<vector<vector<vector<double>>>> temp;
  mult4D(temp, v1, beta1);
  vector<vector<vector<vector<double>>>> temp2;
  mult4D(temp2, df1, 1 - beta1);
  divi4D(temp2, temp2, imageAmount);
  add4D(v1, temp, temp2);

  mult4D(temp, s1, beta2);
  divi4D(temp2, df1, imageAmount);
  mult4D(temp2, temp2, 1 - beta2);
  square4D(temp2, temp2);
  add4D(s1, temp, temp2);

  mult4D(temp, v1, lr);
  addN4D(temp2, s1, .0000001);
  sqrt4D(temp2, temp2);
  divi4DMat(temp, temp, temp2);
  sub4DMat(f1, f1, temp);

  // b1
  vector<vector<double>> temp3;
  mult(temp3, bv1, beta1);
  vector<vector<double>> temp4;
  mult(temp4, db1, 1 - beta1);
  divi2D(temp4, temp4, imageAmount);
  add(bv1, temp3, temp4);

  mult(temp3, bs1, beta2);
  divi2D(temp4, db1, imageAmount);
  mult(temp4, temp4, 1 - beta2);
  square(temp4, temp4);
  add(bs1, temp3, temp4);

  mult(temp3, bv1, lr);
  addN2D(temp4, bs1, .0000001);
  sqrt2D(temp4, temp4);
  divi2DMat(temp3, temp3, temp4);
  sub2DMat(b1, b1, temp3);

  // f2
  mult4D(temp, v2, beta1);
  mult4D(temp2, df2, 1 - beta1);
  divi4D(temp2, temp2, imageAmount);
  add4D(v2, temp, temp2);

  mult4D(temp, s2, beta2);
  divi4D(temp2, df2, imageAmount);
  mult4D(temp2, temp2, 1 - beta2);
  square4D(temp2, temp2);
  add4D(s2, temp, temp2);

  mult4D(temp, v2, lr);
  addN4D(temp2, s2, .0000001);
  sqrt4D(temp2, temp2);
  divi4DMat(temp, temp, temp2);
  sub4DMat(f2, f2, temp);

  // b2
  mult(temp3, bv2, beta1);
  mult(temp4, db2, 1 - beta1);
  divi2D(temp4, temp4, imageAmount);
  add(bv2, temp3, temp4);

  mult(temp3, bs2, beta2);
  divi2D(temp4, db2, imageAmount);
  mult(temp4, temp4, 1 - beta2);
  square(temp4, temp4);
  add(bs2, temp3, temp4);

  mult(temp3, bv2, lr);
  addN2D(temp4, bs2, .0000001);
  sqrt2D(temp4, temp4);
  divi2DMat(temp3, temp3, temp4);
  sub2DMat(b2, b2, temp3);

  // w3
  mult(temp3, v3, beta1);
  mult(temp4, dw3, 1 - beta1);
  divi2D(temp4, temp4, imageAmount);
  add(v3, temp3, temp4);

  mult(temp3, s3, beta2);
  divi2D(temp4, dw3, imageAmount);
  mult(temp4, temp4, 1 - beta2);
  square(temp4, temp4);
  add(s3, temp3, temp4);

  mult(temp3, v3, lr);
  addN2D(temp4, s3, .0000001);
  sqrt2D(temp4, temp4);
  divi2DMat(temp3, temp3, temp4);
  sub2DMat(w3, w3, temp3);

  // b3
  mult(temp3, bv3, beta1);
  mult(temp4, db3, 1 - beta1);
  divi2D(temp4, temp4, imageAmount);
  add(bv3, temp3, temp4);

  mult(temp3, bs3, beta2);
  divi2D(temp4, db3, imageAmount);
  mult(temp4, temp4, 1 - beta2);
  square(temp4, temp4);
  add(bs3, temp3, temp4);

  mult(temp3, bv3, lr);
  addN2D(temp4, bs3, .0000001);
  sqrt2D(temp4, temp4);
  divi2DMat(temp3, temp3, temp4);
  sub2DMat(b3, b3, temp3);

  // w4
  mult(temp3, v4, beta1);
  mult(temp4, dw4, 1 - beta1);
  divi2D(temp4, temp4, imageAmount);
  add(v4, temp3, temp4);

  mult(temp3, s4, beta2);
  divi2D(temp4, dw4, imageAmount);
  mult(temp4, temp4, 1 - beta2);
  square(temp4, temp4);
  add(s4, temp3, temp4);

  mult(temp3, v4, lr);
  addN2D(temp4, s4, .0000001);
  sqrt2D(temp4, temp4);
  divi2DMat(temp3, temp3, temp4);
  sub2DMat(w4, w4, temp3);

  // b4
  mult(temp3, bv4, beta1);
  mult(temp4, db4, 1 - beta1);
  divi2D(temp4, temp4, imageAmount);
  add(bv4, temp3, temp4);

  mult(temp3, bs4, beta2);
  divi2D(temp4, db4, imageAmount);
  mult(temp4, temp4, 1 - beta2);
  square(temp4, temp4);
  add(bs4, temp3, temp4);

  mult(temp3, bv4, lr);
  addN2D(temp4, bs4, .0000001);
  sqrt2D(temp4, temp4);
  divi2DMat(temp3, temp3, temp4);
  sub2DMat(b4, b4, temp3);

  _cost /= imageAmount;
  cost.push_back(_cost);
}

void CNN::conv(double &_loss, vector<vector<vector<vector<double>>>> &_df1, vector<vector<vector<vector<double>>>> &_df2, vector<vector<double>> &_dw3, vector<vector<double>> &_dw4, vector<vector<double>> &_db1, vector<vector<double>> &_db2, vector<vector<double>> &_db3, vector<vector<double>> &_db4, vector<vector<vector<double>>> image, vector<vector<double>> label)
{
  // Forward
  vector<vector<vector<double>>> conv1;
  convolution(conv1, image, f1, b1);
  ReLU(conv1);
  vector<vector<vector<double>>> conv2;
  convolution(conv2, conv1, f2, b2);
  ReLU(conv2);
  vector<vector<vector<double>>> pooled;
  maxpool(pooled, conv2);
  // Fully connected
  vector<vector<double>> flat;
  for (auto &row : pooled)
  {
    for (auto &col : row)
    {
      for (auto &ele : col)
      {
        flat.push_back({ele});
      }
    }
  }
  vector<vector<double>> z;
  dot(z, w3, flat);
  add(z, z, b3);
  ReLU2D(z);
  vector<vector<double>> out;
  dot(out, w4, z);
  add(out, out, b4);
  vector<vector<double>> probs;
  softmax(probs, out);
  double loss;
  categoricalCrossEntropy(loss, probs, label);
  // Backward
  vector<vector<double>> dout;
  sub(dout, probs, label);
  vector<vector<double>> zT;
  transpose(zT, z);
  vector<vector<double>> dW4;
  dot(dW4, dout, zT);
  vector<vector<double>> db4;
  sum(db4, dout, 1);
  vector<vector<double>> w4T;
  transpose(w4T, w4);
  vector<vector<double>> dz;
  dot(dz, w4T, dout);
  ReLU2D(dz);
  vector<vector<double>> flatT;
  transpose(flatT, flat);
  vector<vector<double>> dW3;
  dot(dW3, dz, flatT);
  vector<vector<double>> db3;
  sum(db3, dz, 1);
  vector<vector<double>> w3T;
  transpose(w3T, w3);
  vector<vector<double>> dFlat;
  dot(dFlat, w3T, dz);
  // Conv Backward
  vector<vector<vector<double>>> dpool;
  int index = 0;
  for (int i = 0; i < 8; i++)
  {
    vector<vector<double>> temp1;
    for (int j = 0; j < 10; j++)
    {
      vector<double> temp2;
      for (int k = 0; k < 10; k++)
      {
        temp2.push_back(dFlat[index][0]);
        index++;
      }
      temp1.push_back({temp2});
    }
    dpool.push_back({temp1});
  }
  vector<vector<vector<double>>> dconv2;
  maxpoolBackward(dconv2, dpool, conv2);
  vector<vector<vector<double>>> dconv1;
  vector<vector<vector<vector<double>>>> df2;
  vector<vector<double>> db2;
  convolutionBackward(dconv1, df2, db2, dconv2, conv1, f2);
  vector<vector<vector<double>>> _;
  vector<vector<vector<vector<double>>>> df1;
  vector<vector<double>> db1;
  convolutionBackward(_, df1, db1, dconv1, image, f1);

  _loss = loss;
  _df1 = df1;
  _df2 = df2;
  _dw3 = dW3;
  _dw4 = dW4;
  _db1 = db1;
  _db2 = db2;
  _db3 = db3;
  _db4 = db4;
}

void CNN::predict(vector<vector<double>>  &_probs, vector<vector<vector<double>>> image)
{
  vector<vector<vector<double>>> conv1;
  convolution(conv1, image, f1, b1);
  ReLU(conv1);
  vector<vector<vector<double>>> conv2;
  convolution(conv2, conv1, f2, b2);
  ReLU(conv2);
  vector<vector<vector<double>>> pooled;
  maxpool(pooled, conv2);
  vector<vector<double>> flat;
  for (auto &row : pooled)
  {
    for (auto &col : row)
    {
      for (auto &ele : col)
      {
        flat.push_back({ele});
      }
    }
  }
  vector<vector<double>> z;
  dot(z, w3, flat);
  add(z, z, b3);
  ReLU2D(z);
  vector<vector<double>> out;
  dot(out, w4, z);
  add(out, out, b4);
  vector<vector<double>> probs;
  softmax(probs, out);
  _probs = probs;
}

CNN::CNN()
{
  params.push_back({8, 1, 5, 5}); // f1
  params.push_back({8, 8, 5, 5}); // f2
  params.push_back({128, 800});   // w3
  params.push_back({10, 128});    // w4
  params.push_back({8, 1});       // b1
  params.push_back({8, 1});       // b2
  params.push_back({128, 1});     // b3
  params.push_back({10, 1});      // b4
  f1 = vector<vector<vector<vector<double>>>>(params[0][0], vector<vector<vector<double>>>(params[0][1], vector<vector<double>>(params[0][2], vector<double>(params[0][3], 0))));
  f2 = vector<vector<vector<vector<double>>>>(params[1][0], vector<vector<vector<double>>>(params[1][1], vector<vector<double>>(params[1][2], vector<double>(params[1][3], 0))));
  w3 = vector<vector<double>>(params[2][0], vector<double>(params[2][1], 0));
  w4 = vector<vector<double>>(params[3][0], vector<double>(params[3][1], 0));
  b1 = vector<vector<double>>(params[4][0], vector<double>(params[4][1], 0));
  b2 = vector<vector<double>>(params[5][0], vector<double>(params[5][1], 0));
  b3 = vector<vector<double>>(params[6][0], vector<double>(params[6][1], 0));
  b4 = vector<vector<double>>(params[7][0], vector<double>(params[7][1], 0));
  initializeParameters();
}

void CNN::importData(string fileName)
{
  for (int i = 0; i < 8; i++)
  {
    vector<string> read;
    getRow(read, fileName, i);
    int index = 0;
    if (i == 0)
    {
      for (int i = 0; i < f1.size(); i++)
      {
        for (int j = 0; j < f1[0].size(); j++)
        {
          for (int k = 0; k < f1[0][0].size(); k++)
          {
            for (int l = 0; l < f1[0][0][0].size(); l++)
            {
              f1[i][j][k][l] = stod(read[index]);
              index++;
            }
          }
        }
      }
    }
    else if (i == 1)
    {
      for (int i = 0; i < f2.size(); i++)
      {
        for (int j = 0; j < f2[0].size(); j++)
        {
          for (int k = 0; k < f2[0][0].size(); k++)
          {
            for (int l = 0; l < f2[0][0][0].size(); l++)
            {
              f2[i][j][k][l] = stod(read[index]);
              index++;
            }
          }
        }
      }
    }
    else if (i == 2)
    {
      for (int i = 0; i < w3.size(); i++)
      {
        for (int j = 0; j < w3[0].size(); j++)
        {
          w3[i][j] = stod(read[index]);
          index++;
        }
      }
    }
    else if (i == 3)
    {
      for (int i = 0; i < w4.size(); i++)
      {
        for (int j = 0; j < w4[0].size(); j++)
        {
          w4[i][j] = stod(read[index]);
          index++;
        }
      }
    }
    else if (i == 4)
    {
      for (int i = 0; i < b1.size(); i++)
      {
        for (int j = 0; j < b1[0].size(); j++)
        {
          b1[i][j] = stod(read[index]);
          index++;
        }
      }
    }
    else if (i == 5)
    {
      for (int i = 0; i < b2.size(); i++)
      {
        for (int j = 0; j < b2[0].size(); j++)
        {
          b2[i][j] = stod(read[index]);
          index++;
        }
      }
    }
    else if (i == 6)
    {
      for (int i = 0; i < b3.size(); i++)
      {
        for (int j = 0; j < b3[0].size(); j++)
        {
          b3[i][j] = stod(read[index]);
          index++;
        }
      }
    }
    else if (i == 7)
    {
      for (int i = 0; i < b4.size(); i++)
      {
        for (int j = 0; j < b4[0].size(); j++)
        {
          b4[i][j] = stod(read[index]);
          index++;
        }
      }
    }
  }
}

void CNN::exportData(string fileName)
{
  remove(fileName.c_str());
  ofstream file;
  file.open(fileName);
  file << "\n";
  for (int i = 0; i < f1.size(); i++)
  {
    for (int j = 0; j < f1[0].size(); j++)
    {
      for (int k = 0; k < f1[0][0].size(); k++)
      {
        for (int l = 0; l < f1[0][0][0].size(); l++)
        {
          file << f1[i][j][k][l] << ",";
        }
      }
    }
  }
  file << "\n";
  for (int i = 0; i < f2.size(); i++)
  {
    for (int j = 0; j < f2[0].size(); j++)
    {
      for (int k = 0; k < f2[0][0].size(); k++)
      {
        for (int l = 0; l < f2[0][0][0].size(); l++)
        {
          file << f2[i][j][k][l] << ",";
        }
      }
    }
  }
  file << "\n";
  for (int i = 0; i < w3.size(); i++)
  {
    for (int j = 0; j < w3[0].size(); j++)
    {
      file << w3[i][j] << ",";
    }
  }
  file << "\n";
  for (int i = 0; i < w4.size(); i++)
  {
    for (int j = 0; j < w4[0].size(); j++)
    {
      file << w4[i][j] << ",";
    }
  }
  file << "\n";
  for (int i = 0; i < b1.size(); i++)
  {
    for (int j = 0; j < b1[0].size(); j++)
    {
      file << b1[i][j] << ",";
    }
  }
  file << "\n";
  for (int i = 0; i < b2.size(); i++)
  {
    for (int j = 0; j < b2[0].size(); j++)
    {
      file << b2[i][j] << ",";
    }
  }
  file << "\n";
  for (int i = 0; i < b3.size(); i++)
  {
    for (int j = 0; j < b3[0].size(); j++)
    {
      file << b3[i][j] << ",";
    }
  }
  file << "\n";
  for (int i = 0; i < b4.size(); i++)
  {
    for (int j = 0; j < b4[0].size(); j++)
    {
      file << b4[i][j] << ",";
    }
  }
  file.close();
}

void CNN::initializeParameters()
{
  srand(time(NULL));
  double dev1 = 1 / sqrt(f1.size() * f1[0].size() * f1[0][0].size() * f1[0][0][0].size());
  double dev2 = 1 / sqrt(f2.size() * f2[0].size() * f2[0][0].size() * f2[0][0][0].size());
  for (int i = 0; i < f1.size(); i++)
  {
    for (int j = 0; j < f1[0].size(); j++)
    {
      for (int k = 0; k < f1[0][0].size(); k++)
      {
        for (int l = 0; l < f1[0][0][0].size(); l++)
        {
          double f = (double)rand() / RAND_MAX;
          double r = f * 4 - 2;
          f1[i][j][k][l] = dev1 * exp(-1 * ((r * r) / 2));
          if (((double)rand() / (RAND_MAX)) > 0.5)
          {
            f1[i][j][k][l] *= -1;
          }
          else
          {
            f1[i][j][k][l] *= 1;
          }
        }
      }
    }
  }
  for (int i = 0; i < f2.size(); i++)
  {
    for (int j = 0; j < f2[0].size(); j++)
    {
      for (int k = 0; k < f2[0][0].size(); k++)
      {
        for (int l = 0; l < f2[0][0][0].size(); l++)
        {
          double f = (double)rand() / RAND_MAX;
          double r = f * 6 - 3;
          f2[i][j][k][l] = dev2 * exp(-1 * ((r * r) / 2));
          if (((double)rand() / (RAND_MAX)) > 0.5)
          {
            f2[i][j][k][l] *= -1;
          }
          else
          {
            f2[i][j][k][l] *= 1;
          }
        }
      }
    }
  }
  for (int i = 0; i < w3.size(); i++)
  {
    for (int j = 0; j < w3[0].size(); j++)
    {
      w3[i][j] = randGaussian() * 0.01;
    }
  }
  for (int i = 0; i < w4.size(); i++)
  {
    for (int j = 0; j < w4[0].size(); j++)
    {
      w4[i][j] = randGaussian() * 0.01;
    }
  }
}

double CNN::randGaussian()
{
  double r = ((double)rand() / (RAND_MAX));
  return min(1.0, (((double)rand() / (RAND_MAX)) > 0.5 ? sqrt(-2 * log(r + .05)) : -1 * sqrt(-2 * log(r + .05))));
  // double f = (double)rand() / RAND_MAX;
  // double r = f * 6 - 3;
  // return exp(-1 * ((r * r) / 2));
}

void CNN::getMNISTData(vector<vector<vector<double>>> &d, int &l, int rowNum, string fileName)
{
  vector<string> read;
  getRow(read, fileName, rowNum);
  l = stoi(read[0]);
  vector<vector<double>> mnist1;
  int index = 1;
  for (int i = 0; i < 28; i++)
  {
    vector<double> temp;
    for (int j = 0; j < 28; j++)
    {
      temp.push_back(stoi(read[index]));
      index++;
    }
    mnist1.push_back(temp);
  }
  double mean1;
  meanAll(mean1, mnist1);
  int realMean = (int)mean1;
  vector<vector<double>> result(28, vector<double>(28, 0));
  for (int i = 0; i < 28; i++)
  {
    for (int j = 0; j < 28; j++)
    {
      result[i][j] = mnist1[i][j] - realMean;
    }
  }
  double stdVal;
  stdAll(stdVal, result);
  int intSTD = (int)stdVal;
  for (int i = 0; i < 28; i++)
  {
    for (int j = 0; j < 28; j++)
    {
      result[i][j] = result[i][j] / intSTD;
    }
  }
  d.push_back(result);
}