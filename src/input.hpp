#ifndef INPUT_H
#define INPUT_H

#include <string>
#include <fstream>
#include <iostream>
#include <stdint.h>

const int MAXN  = 6e4 + 1;
extern unsigned int image[MAXN][28][28];
extern unsigned int num, magic, rows, cols;
extern unsigned int label[MAXN];

unsigned int in(std::ifstream& icin, unsigned int size);
void input();
void display_image(unsigned int *img[28][28]);

#endif
