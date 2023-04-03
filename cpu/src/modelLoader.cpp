#include "modelLoader.h"

char *fileName;  // for IsEqual function

void SetNull(char *buf, int size) {
  for (int i = 0; i < size; i++) {
    buf[i] = '\0';
  }
}

void EraseString(char *buf, int startIdx, int endIdx) {
  int n = endIdx + 1 - startIdx;
  int idx;

  for (idx = endIdx + 1; buf[idx] != '\0'; idx++) {
    buf[idx - n] = buf[idx];
  }
  buf[idx - n] = buf[idx];
}

bool IsEqual(const pair<char *, float *> &element) {
  return !strcmp(element.first, fileName);
}

File::File(unsigned int startOffset, char *name, unsigned short nameLen,
           unsigned int dataSize, unsigned short extraFieldLen) {
  startOffset_ = startOffset;
  name_ = name;
  nameLen_ = nameLen;
  dataSize_ = dataSize;
  extraFieldLen_ = extraFieldLen;
}

void ModelLoader::FseekFileData(File *file) {
  char pkLabel[5] = "";
  unsigned short extraFieldLen;

  // cout << "file start_offset : " << data_file->start_offset << endl;
  // PK label
  fseek(fp, file->startOffset_, SEEK_SET);
  fread(pkLabel, 4, 1, fp);
  // cout << "local file header : " << pkLabel << endl;

  // extra_field_len
  fseek(fp, 24, SEEK_CUR);
  fread(&extraFieldLen, 2, 1, fp);
  file->extraFieldLen_ = extraFieldLen;
  // cout << "file extra field len : " << file->extraFieldLen_ << endl;
  // cout << "name len : " << file->nameLen_ << endl;

  // file contents
  fseek(fp, file->nameLen_ + file->extraFieldLen_, SEEK_CUR);
}

File::~File() { free(name_); }

ModelLoader::ModelLoader(char *filePath) {
  fp = fopen(filePath, "rb");
  if (fp == NULL) {
    cerr << "ModelLoader open fail..." << endl;
    exit(-1);
  }

  char *buf;
  char pkLabel[5] = "";
  unsigned short fileCount;
  unsigned int startOffset;
  unsigned short fileNameLen;
  unsigned int dataSize;
  unsigned short extraFieldLen, k;

  // PK56 : End of central directory record (EOCD)
  fseek(fp, -22, SEEK_END);
  fread(pkLabel, 4, 1, fp);
  // cout << pkLabel << endl;

  // pass 4 byte, read 2 byte : total number of files
  // files : data.pkl + weightfiles + version
  fseek(fp, 4, SEEK_CUR);
  fread(&fileCount, 2, 1, fp);
  // cout << "file count  : " << fileCount << endl << endl;

  // pass 6 byte, read 4 byte : startOffset
  fseek(fp, 6, SEEK_CUR);
  fread(&startOffset, 4, 1, fp);

  // seek start of central directory
  fseek(fp, startOffset, SEEK_SET);

  for (int i = 0; i < fileCount; i++) {
    // PKXX lobal
    fread(pkLabel, 4, 1, fp);
    // cout << pkLabel << endl;

    // compressed_size  ----> datasize
    fseek(fp, 16, SEEK_CUR);
    fread(&dataSize, 4, 1, fp);
    cout << "data size : " << dataSize << endl;

    // file_name_len, m, k
    fseek(fp, 4, SEEK_CUR);
    fread(&fileNameLen, 2, 1, fp);
    fread(&extraFieldLen, 2, 1, fp);
    // cout << "extraFieldLen: " << extraFieldLen << endl;
    fread(&k, 2, 1, fp);
    // cout << "file_name_len : " << file_name_len << endl;

    // local_file_header_offset --> file start offset
    fseek(fp, 8, SEEK_CUR);
    fread(&startOffset, 4, 1, fp);

    // file_name
    buf = new char[fileNameLen + 1];
    SetNull(buf, fileNameLen + 1);
    fread(buf, fileNameLen, 1, fp);

    cout << buf << endl << endl;

    fseek(fp, extraFieldLen + k, SEEK_CUR);

    // add File to list
    File *file = new File(startOffset, buf, fileNameLen, dataSize, 0);
    fileList.push_back(file);
  }

  cout << "number of files : " << fileList.size() << endl;

  ParsePickleFile();
  ParseWeightFile();

  fclose(fp);
}

void ModelLoader::ParsePickleFile() {
  File *file = fileList.front();
  FseekFileData(file);  // move offset to file data

  bool isFileName = false;
  char *name;
  unsigned int nameLen;
  char c;
  while (1) {
    // find 'X' character
    fread(&c, 1, 1, fp);

    if (c == 'X') {
      // param name len
      fread(&nameLen, 4, 1, fp);
      // pass exception
      if (nameLen > 100) continue;

      name = new char[nameLen + 1];
      SetNull(name, nameLen + 1);
      fread(name, nameLen, 1, fp);

      if (!strcmp(name, "cpu") || !strcmp(name, "storage"))
        continue;
      else if (!strcmp(name, "_metadata"))
        break;

      cout << "name : " << name << endl;
      if (isFileName) {
        // cout << "filename : " << name << endl;
        weightList.push_back(make_pair(name, (float *)NULL));
      }

      // change the state ( param_name or file_name)
      isFileName = !isFileName;
    }
  }
  cout << "wieght list size : " << weightList.size() << endl;

  // delete pickle, version file
  fileList.pop_front();
  fileList.pop_back();
}

void ModelLoader::ParseWeightFile() {
  list<File *>::iterator iter;
  float *data;
  int dataSize;
  for (iter = fileList.begin(); iter != fileList.end(); iter++) {
    File *file = *iter;
    EraseString(file->name_, 0, 12);
    fileName = file->name_;

    auto it = find_if(weightList.begin(), weightList.end(), IsEqual);
    if (it == weightList.end()) assert("cannot find weight");

    // set data
    FseekFileData(file);  // move offset to file data
    dataSize = file->dataSize_;

    data = new float[dataSize / 4];
    fread(data, dataSize, 1, fp);
    it->second = data;
  }
}

float *ModelLoader::GetWeight() {
  pair<char *, float *> element = weightList.front();
  cout << "name : " << element.first << endl;

  /*
  float* ptr = element.second;

  for( int i = 0; i < 10; i++ )
  cout << "data : " << ptr[i] << endl;
  cout << endl;
  */

  // float *weight = weightList.front().second;
  float *weight = element.second;
  weightList.pop_front();

  return weight;
}

/*
int main() {
        ModelLoader modelLoader("../../pth_data/pretrain_model.pth");
        for( int i = 0; i < 2; i++ ) {
                float* weight = modelLoader.GetWeight();
                for( int i = 0; i < 10; i++ )
                        cout << "data : " << weight[i] << endl;
                cout << endl;
        }

}
*/
