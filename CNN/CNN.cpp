#include "CNN.h"
#include "../utils/MatrixFunctions.h"
#include "Forward.h"
#include "Backward.h"
#include "../common.h"

void CNN::train(int epochs, int imageAmount, vector<vector<vector<vector<double>>>> images, vector<int> labels)
{
  vector<double> costs;
  for (int i = 0; i < epochs; i++)
  {
    adamGD(imageAmount, images, labels, costs);
  }
}

void CNN::adamGD(int imageAmount, vector<vector<vector<vector<double>>>> images, vector<int> labels, vector<double> &cost)
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
    vector<vector<double>> label(10, vector<double>(1, 0));
    label[labels[i]][0] = 1;
    double loss;
    vector<vector<vector<vector<double>>>> _df1;
    vector<vector<vector<vector<double>>>> _df2;
    vector<vector<double>> _dw3;
    vector<vector<double>> _dw4;
    vector<vector<double>> _db1;
    vector<vector<double>> _db2;
    vector<vector<double>> _db3;
    vector<vector<double>> _db4;
    conv(loss, _df1, _df2, _dw3, _dw4, _db1, _db2, _db3, _db4, images[i], label);
    cout << loss << endl;
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

  // for (auto &row : df1)
  // {
  //   for (auto &col : row)
  //   {
  //     for (auto &ele : col)
  //     {
  //       for (auto &ele2 : ele)
  //       {
  //         cout << ele2 << " ";
  //       }
  //       cout << endl;
  //     }
  //   }
  //   cout << endl;
  // }
}

CNN::CNN(vector<vector<int>> _params)
{
  params = _params;
  f1 = vector<vector<vector<vector<double>>>>(_params[0][0], vector<vector<vector<double>>>(_params[0][1], vector<vector<double>>(_params[0][2], vector<double>(_params[0][3], 1))));
  b1 = vector<vector<double>>(_params[4][0], vector<double>(_params[4][1], 0));
  f2 = vector<vector<vector<vector<double>>>>(_params[1][0], vector<vector<vector<double>>>(_params[1][1], vector<vector<double>>(_params[1][2], vector<double>(_params[1][3], 1))));
  b2 = vector<vector<double>>(_params[5][0], vector<double>(_params[5][1], 0));
  w3 = vector<vector<double>>(_params[2][0], vector<double>(_params[2][1], .00001));
  b3 = vector<vector<double>>(_params[6][0], vector<double>(_params[6][1], 0));
  w4 = vector<vector<double>>(_params[3][0], vector<double>(_params[3][1], .00001));
  b4 = vector<vector<double>>(_params[7][0], vector<double>(_params[7][1], 0));
}

double CNN::randGaussian()
{
  double r = ((double)rand() / (RAND_MAX));
  return ((double)rand() / (RAND_MAX)) > 0.5 ? sqrt(-2 * log(r)) : -1 * sqrt(-2 * log(r));
}