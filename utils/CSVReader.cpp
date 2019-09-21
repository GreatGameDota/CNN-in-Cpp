#include <fstream>
#include <sstream>
#include <string>
#include "../common.h"

void getRow(vector<string> &result, string fileName, int rowNum)
{
  fstream fin;
  fin.open("csv/" + fileName + ".csv", ios::in);
  int rollnum = rowNum, roll2, count = 0;
  vector<string> row;
  string line, word, temp;
  getline(fin, line);
  int amount = 0;
  while (amount < 60000)
  {
    row.clear();
    getline(fin, line);
    if (amount == rowNum)
    {
      stringstream s(line);
      while (getline(s, word, ','))
      {
        row.push_back(word);
      }
      count = 1;
      break;
    }
    amount++;
  }
  if (count == 0)
    cout << "Record not found\n";
  result = row;
  fin.close();
}