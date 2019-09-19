#include "Backward.h"
#include "../utils/MatrixFunctions.h"
#include "../common.h"

void convolutionBackward(vector<vector<vector<double>>> &result, vector<vector<vector<double>>> dconv_prev, vector<vector<vector<double>>> conv_in, vector<vector<vector<vector<double>>>> filter, int stride = 1)
{
  
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