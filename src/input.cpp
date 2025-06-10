#include "input.hpp"

unsigned int image[MAXN][28][28];
unsigned int num, magic, rows, cols;
unsigned int label[MAXN];

unsigned int in(std::ifstream& icin, unsigned int size) {
    unsigned int ans = 0;
    for (int i = 0; i < size; i++) {
        unsigned char x;
        icin.read((char*)&x, 1);
        unsigned int temp = x;
        ans <<= 8;
        ans += temp;
    }
    return ans;
}

void input() {
    std::ifstream icin;
    icin.open("data/train-images.idx3-ubyte", std::ios::binary);
    magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);
    for (int i = 0; i < num; i++) {
        for (int x = 0; x < rows; x++) {
            for (int y = 0; y < cols; y++) {
                image[i][x][y] = in(icin, 1);
            }
        }
    }
    icin.close();
    icin.open("data/train-labels.idx1-ubyte", std::ios::binary);
    magic = in(icin, 4), num = in(icin, 4);
    for (int i = 0; i < num; i++) {
        label[i] = in(icin, 1);
    }
    icin.close();
}

void display_image(unsigned int *img[28][28]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if ((*img)[i][j] > 50 && (*img)[i][j] < 150) {
                std::cout << "s";
            }
            if ((*img)[i][j] >= 150) {
                std::cout << "S";
            }
            if ((*img)[i][j] <= 50) {
                std::cout << ".";
            }
        }
        std::cout << '\n';
    }
}
