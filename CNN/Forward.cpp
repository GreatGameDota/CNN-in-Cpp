#include "Forward.h"
#include "../utils/MatrixFunctions.h"
#include "../common.h"

void convolution(vector<vector<vector<double>>> &result, vector<vector<vector<double>>> image, vector<vector<vector<vector<double>>>> filter, vector<vector<double>> bias, int stride)
{
  int filt1 = filter.size();
  int filt2 = filter[0].size();
  int filt3 = filter[0][0].size();
  int img1 = image.size();
  int img2 = image[0].size();
  int img3 = image[0][0].size();
  int out_dim = (int)((img2 - filt3) / stride) + 1;
  // cout << out_dim << endl;
  if (img1 != filt2)
  {
    cout << "ERROR: Dimensions of Filter and Image must match!" << endl;
  }
  vector<vector<vector<double>>> res(filt1, vector<vector<double>>(out_dim, vector<double>(out_dim, 0)));
  for (int curr_f = 0; curr_f < filt1; curr_f++)
  {
    int curr_y = 0;
    int out_y = 0;
    while (curr_y + filt3 <= img2)
    {
      int curr_x = 0;
      int out_x = 0;
      while (curr_x + filt3 <= img2)
      {
        vector<vector<vector<double>>> imageSection;
        for (int k = 0; k < filt2; k++)
        {
          vector<vector<double>> temp2;
          for (int i = curr_y; i < curr_y + filt3; i++)
          {
            vector<double> temp;
            for (int j = curr_x; j < curr_x + filt3; j++)
            {
              temp.push_back(image[k][i][j]);
            }
            temp2.push_back(temp);
          }
          imageSection.push_back({temp2});
        }
        vector<vector<vector<double>>> res2;
        multMatrices3D(res2, filter[curr_f], imageSection);
        sum3D(res2, res2, 0);
        res2[0][0][0] += bias[curr_f][0];
        res[curr_f][out_y][out_x] = res2[0][0][0];

        curr_x += stride;
        out_x++;
      }
      curr_y += stride;
      out_y++;
    }
  }
  result = res;
}

void ReLU(vector<vector<vector<double>>> &result)
{
  for (int i = 0; i < result.size(); i++)
  {
    for (int j = 0; j < result[0].size(); j++)
    {
      for (int k = 0; k < result[0][0].size(); k++)
      {
        if (result[i][j][k] < 0)
        {
          result[i][j][k] = 0;
        }
      }
    }
  }
}

void maxpool(vector<vector<vector<double>>> &result, vector<vector<vector<double>>> image, int size, int stride)
{
  int img1 = image.size();
  int img2 = image[0].size();
  int img3 = image[0][0].size();
  int w = (int)((img2 - size) / stride) + 1;
  int h = (int)((img3 - size) / stride) + 1;
  vector<vector<vector<double>>> res(img1, vector<vector<double>>(h, vector<double>(w, 0)));
  for (int i = 0; i < img1; i++)
  {
    int curr_y = 0;
    int out_y = 0;
    while (curr_y + size <= img2)
    {
      int curr_x = 0;
      int out_x = 0;
      while (curr_x + size <= img2)
      {
        vector<vector<vector<double>>> imageSection;
        for (int k = 0; k < img1; k++)
        {
          vector<vector<double>> temp2;
          for (int i = curr_y; i < curr_y + size; i++)
          {
            vector<double> temp;
            for (int j = curr_x; j < curr_x + size; j++)
            {
              temp.push_back(image[k][i][j]);
            }
            temp2.push_back(temp);
          }
          imageSection.push_back({temp2});
        }
        double max = imageSection[0][0][0];
        for (int k = 0; k < imageSection[0][0].size(); k++)
        {
          for (int i = 1; i < imageSection.size(); i++)
          {
            for (int j = 0; j < imageSection[0].size(); j++)
            {
              if (imageSection[i][j][k] > max)
                max = imageSection[i][j][k];
            }
          }
        }
        res[i][out_y][out_x] = max;

        curr_x += stride;
        out_x++;
      }
      curr_y += stride;
      out_y++;
    }
  }
  result = res;
}

void softmax(vector<vector<double>> &result, vector<vector<double>> X)
{
  int row = X.size();
  int col = X[0].size();
  vector<vector<double>> res(row, vector<double>(col, 0));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      res[i][j] = exp(X[i][j]);
    }
  }
  vector<vector<double>> sumAll;
  sum(sumAll, res, 0);
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      res[i][j] /= sumAll[0][0];
    }
  }
  result = res;
}

void categoricalCrossEntropy(double &result, vector<vector<double>> probs, vector<vector<double>> label)
{
  int row = probs.size();
  int col = probs[0].size();
  vector<vector<double>> temp(row, vector<double>(col, 0));
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      temp[i][j] = log(probs[i][j]);
    }
  }
  multMatrices(temp, temp, label);
  sum(temp, temp, 0);
  result = -1 * temp[0][0];
}