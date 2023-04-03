

#ifndef MODELLOADER_H_
#define MODELLOADER_H_

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <list>

using namespace std;

bool IsEqual(const pair<char*, float*>& element);
void EraseString(char* buf, int startIdx, int endIdx);
void SetNull(char* buf, int size);

class File {
 public:
  unsigned int startOffset_;
  char* name_;
  unsigned short nameLen_;
  unsigned int dataSize_;
  unsigned short extraFieldLen_;

 public:
  File(unsigned int startOffset, char* name, unsigned short nameLen,
       unsigned int dataSize, unsigned short extraFieldLen);
  ~File();
};

class ModelLoader {
 public:
  ModelLoader(char* filePath);  // pth file
  void ParsePickleFile();
  void ParseWeightFile();
  float* GetWeight();
  void FseekFileData(File* file);

 private:
  FILE* fp;
  list<File*> fileList;
  list<pair<char*, float*>> weightList;  // pair<filename, weight>
};

#endif
