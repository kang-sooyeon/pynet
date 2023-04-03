
#ifndef UTIL_H_
#define UTIL_H_

#include <iostream>

#include "memory.h"

using namespace std;

void Print(char* layerName, cpu::Memory* m);
void checkError(string path, cpu::Memory* m);
void SavePng(float* array, size_t size, int idx, int H, int W);

static double get_time();
void timer_start(int i);
double timer_stop(int i);

#endif
