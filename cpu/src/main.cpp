
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "../include/cpu.h"

using namespace std;

size_t largeSize = 1650000000;
size_t smallSize = 769205248;

int main(int argc, char *argv[]) {
  string datasetDir = argv[1];
  int size = atoi(argv[3]);
  float scale = 1;
  int level = 0;
  bool fullResolution = true;

  cpu::Device *dev = new cpu::Device();

  ///////////////////////////// buffer
  // large buffer
  cpu::Memory *l1 = new cpu::Memory(dev, largeSize);
  cpu::Memory *l2 = new cpu::Memory(dev, largeSize);
  cpu::Memory *l3 = new cpu::Memory(dev, largeSize);
  // small buffer
  cpu::Memory *s1 = new cpu::Memory(dev, smallSize);
  cpu::Memory *s2 = new cpu::Memory(dev, smallSize);
  /////////////////////////////

  DataLoader visualData(dev, datasetDir, size, scale, level, fullResolution);
  cout << "visual dataset count : " << visualData.Size() << endl;

  int c, h, w;
  visualData.GetDataSize(&c, &h, &w);

  timer_start(0);
  cpu::op::PyNET pynet(dev, c, h, w, level, true, true);
  cout << "create pynet time : " << timer_stop(0) << " sec" << endl;

  pynet.LoadWeight(argv[2]);
  cout << "Load complete..." << endl;

  for (int i = 0; i < visualData.Size(); i++) {
    timer_start(0);
    visualData.Get(i, l1, l2);
    cout << "load png image : " << timer_stop(0) << " sec" << endl;

    Print("image", l2);

    // inference
    timer_start(0);
    pynet.Forward(l2, l1, l3, s1, s2);
    cout << "inference time : " << timer_stop(0) << " sec" << endl;
    Print("pynet output", s1);

    // save png
    int h, w, c;
    pynet.GetOutputSize(c, h, w);
    timer_start(0);
    SavePng((float *)s1->Ptr(), s1->Size(), i, h, w);
    cout << "Save png time : " << timer_stop(0) << " sec" << endl;

    cout << i << " iter" << endl;
  }

  delete s1;
  delete s2;
  delete l1;
  delete l2;
  delete l3;
}
