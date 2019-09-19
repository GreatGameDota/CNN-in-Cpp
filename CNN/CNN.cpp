#include "CNN.h"
#include "../utils/MatrixFunctions.h"
#include "Forward.h"
#include "Backward.h"
#include "../common.h"

void CNN::adamGD(int imageAmount, vector<vector<vector<vector<double>>>> images, vector<int> labels)
{
  double _cost = 0;
  for (int i = 0; i < imageAmount; i++)
  {
    vector<vector<double>> label(10, vector<double>(1, 0));
    label[labels[i]][0] = 1;

    double loss;
    vector<vector<vector<vector<double>>>> df1;
    vector<vector<vector<vector<double>>>> df2;
    vector<vector<double>> dw3;
    vector<vector<double>> dw4;
    vector<vector<double>> db1;
    vector<vector<double>> db2;
    vector<vector<double>> db3;
    vector<vector<double>> db4;
    conv(loss, df1, df2, dw3, dw4, db1, db2, db3, db4, images[i], label);
    cout << loss << endl;
    _cost += loss;
  }
}

void CNN::conv(double &_loss, vector<vector<vector<vector<double>>>> &_df1, vector<vector<vector<vector<double>>>> &_df2, vector<vector<double>> &_dw3, vector<vector<double>> &_dw4, vector<vector<double>> &_db1, vector<vector<double>> &_db2, vector<vector<double>> &_db3, vector<vector<double>> &_db4, vector<vector<vector<double>>> image, vector<vector<double>> label)
{
  // Forward
  vector<vector<vector<vector<double>>>> filter(8, vector<vector<vector<double>>>(1, vector<vector<double>>(5, vector<double>(5, 1))));
  vector<vector<double>> bias(8, vector<double>(1, 0));
  vector<vector<vector<double>>> conv1;
  convolution(conv1, image, filter, bias);
  ReLU(conv1);
  vector<vector<vector<vector<double>>>> filter2(8, vector<vector<vector<double>>>(8, vector<vector<double>>(5, vector<double>(5, 1))));
  vector<vector<vector<double>>> conv2;
  convolution(conv2, conv1, filter2, bias);
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
  vector<vector<double>> w3(128, vector<double>(800, .00001));
  vector<vector<double>> b3(128, vector<double>(1, 0));
  vector<vector<double>> z;
  dot(z, w3, flat);
  add(z, z, b3);
  for (int i = 0; i < z.size(); i++)
  {
    for (int j = 0; j < z[0].size(); j++)
    {
      if (z[i][j] < 0)
        z[i][j] = 0;
    }
  }
  vector<vector<double>> w4(10, vector<double>(128, .00001));
  vector<vector<double>> b4(10, vector<double>(1, 0));
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
  for (int i = 0; i < dz.size(); i++)
  {
    for (int j = 0; j < dz[0].size(); j++)
    {
      if (dz[i][j] < 0)
        dz[i][j] = 0;
    }
  }
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
  convolutionBackward(dconv1, df2, db2, dconv2, conv1, filter2);
  vector<vector<vector<double>>> _;
  vector<vector<vector<vector<double>>>> df1;
  vector<vector<double>> db1;
  convolutionBackward(_, df1, db1, dconv1, image, filter);

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

double CNN::randGaussian()
{
  double r = ((double)rand() / (RAND_MAX));
  return ((double)rand() / (RAND_MAX)) > 0.5 ? sqrt(-2 * log(r)) : -1 * sqrt(-2 * log(r));
}