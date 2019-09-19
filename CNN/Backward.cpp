#include "Backward.h"
#include "../utils/MatrixFunctions.h"
#include "../common.h"

void convolutionBackward(vector<vector<vector<double>>> &result, vector<vector<vector<vector<double>>>> &df, vector<vector<double>> &db, vector<vector<vector<double>>> dconv_prev, vector<vector<vector<double>>> conv_in, vector<vector<vector<vector<double>>>> filt, int stride)
{
  int filt1 = filt.size();
  int filt2 = filt[0].size();
  int filt3 = filt[0][0].size();
  int orig2 = conv_in[0].size();
  vector<vector<vector<double>>> dout(conv_in.size(), vector<vector<double>>(orig2, vector<double>(conv_in[0][0].size(), 0)));
  vector<vector<vector<vector<double>>>> dfilt(filt1, vector<vector<vector<double>>>(filt2, vector<vector<double>>(filt3, vector<double>(filt[0][0][0].size(), 0))));
  vector<vector<double>> dbias;
  for (int curr_f = 0; curr_f < filt1; curr_f++)
  {
    int curr_y = 0;
    int out_y = 0;
    while (curr_y + filt3 <= orig2)
    {
      int curr_x = 0;
      int out_x = 0;
      while (curr_x + filt3 <= orig2)
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
              temp.push_back(conv_in[k][i][j]);
            }
            temp2.push_back(temp);
          }
          imageSection.push_back({temp2});
        }
        vector<vector<vector<double>>> res;
        mult3D(res, imageSection, dconv_prev[curr_f][out_y][out_x]);
        add3D(res, dfilt[curr_f], res);
        dfilt[curr_f] = res;
        mult3D(res, filt[curr_f], dconv_prev[curr_f][out_y][out_x]);
        for (int k = 0; k < filt2; k++)
        {
          for (int i = curr_y, i2 = 0; i < curr_y + filt3; i++, i2++)
          {
            for (int j = curr_x, j2 = 0; j < curr_x + filt3; j++, j2++)
            {
              dout[k][i][j] += res[k][i2][j2];
            }
          }
        }

        curr_x += stride;
        out_x++;
      }
      curr_y += stride;
      out_y++;
    }
    vector<vector<double>> temp;
    sum(temp, dconv_prev[curr_f], 0);
    dbias.push_back({temp[0][0]});
  }
  result = dout;
  df = dfilt;
  db = dbias;
}

void maxpoolBackward(vector<vector<vector<double>>> &result, vector<vector<vector<double>>> dpool, vector<vector<vector<double>>> orig, int size, int stride)
{
  int orig1 = orig.size();
  int orig2 = orig[0].size();
  vector<vector<vector<double>>> dout(orig1, vector<vector<double>>(orig2, vector<double>(orig[0][0].size(), 0)));
  for (int curr_c = 0; curr_c < orig1; curr_c++)
  {
    int curr_y = 0;
    int out_y = 0;
    while (curr_y + size <= orig2)
    {
      int curr_x = 0;
      int out_x = 0;
      while (curr_x + size <= orig2)
      {
        vector<vector<double>> filterSection;
        for (int i = curr_y; i < curr_y + size; i++)
        {
          vector<double> temp;
          for (int j = curr_x; j < curr_x + size; j++)
          {
            temp.push_back(orig[curr_c][i][j]);
          }
          filterSection.push_back({temp});
        }
        vector<int> maxIndex{0, 0};
        double max = filterSection[0][0];
        for (int i = 1; i < filterSection.size(); i++)
        {
          for (int j = 0; j < filterSection[0].size(); j++)
          {
            if (filterSection[i][j] > max)
            {
              max = filterSection[i][j];
              maxIndex[0] = i;
              maxIndex[1] = j;
            }
          }
        }
        dout[curr_c][curr_y + maxIndex[0]][curr_x + maxIndex[1]] = dpool[curr_c][out_y][out_x];
        
        curr_x += stride;
        out_x++;
      }
      curr_y += stride;
      out_y++;
    }
  }
  result = dout;
}