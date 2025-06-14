#include <iostream>
#include <string>
#include <fstream>
#include <stdint.h>
#include <vector>
#include <stdio.h>
#include <string.h>

using namespace std;

// CONSTANTS START

const int width = 28;
const int height = 28;

// epochs = Number of iterations for back-propagation algorithm
// learning_rate = Learing rate
// momentum = Momentum (heuristics to optimize back-propagation algorithm)
// epsilon = Epsilon, no more iterations if the learning error is smaller than epsilon

const int n_input = width * height; // Number of input neurons (784 pixels of an image)
const int n_hidden = 128; // Number of hidden neurons
const int n_output = 10; // Number of output neurons. Ten classes: 0 - 9
const int n_train_images = 60000; // No. of training images
const int n_test_images = 10000; // No. of images for testing

// no. of neurons in each layers.
const vector<int> n_neurons = {n_input, 10, 10};

const int epochs = 512;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;
// CONSTANTS END

struct Image{
    private:

    public:
        unsigned int val[784];

        void display() {
            char ascii[] = " .,:ilwW";

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int pixel = val[width*i + j];
                    // std::cout << pixel << '\n';
                    cout << (pixel > 100? "0 ":"  ");
                    // std::cout << ascii[(val[pixel])/32];
                }
                std::cout << std::endl;
            }
        }
};

// VARIABLES START
Image image[n_train_images];
unsigned int label[n_train_images]; // Correct label of the training images

Image test_image[n_test_images];
unsigned int test_label[n_test_images]; // the correct label of the test images

vector<vector<vector<double>>> network(n_neurons.size());
// VARIABLES END

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

    unsigned int magic, num, rows, cols; // Metadata. won't be using
    magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);

    for (int i = 0; i < num; i++) {
        for (int x = 0; x < rows; x++) {
            for (int y = 0; y < cols; y++) {
                image[i].val[rows*x + y] = in(icin, 1);
            }
        }
    }
    // for (int j = 0; j < 10; j++) {
    //     for (int i = 0; i < n_input; i++) {
    //         cout << image[j].val[i] << " ";
    //     }
    // }

    icin.close();
    icin.open("data/train-labels.idx1-ubyte", std::ios::binary);
    magic = in(icin, 4), num = in(icin, 4);
    for (int i = 0; i < num; i++) {
        label[i] = in(icin, 1);
    }
    icin.close();
}

void initialize_network() {

}

void load_network() {
    string net_filename = "network/";
    for (int i = 0; i < n_neurons.size(); i++) {
        if (i) { net_filename += '_'; }
        net_filename += to_string(n_neurons[i]);
    }

    ifstream saved_net(net_filename);
    if (saved_net.fail()) {
        initialize_network();
    } else {
        cout << "do something here" << endl;
    }
}

void test() {
    std::ifstream icin;
    icin.open("data/t10k-images.idx3-ubyte", std::ios::binary);

    unsigned int magic, num, rows, cols; // Metadata. won't be using
    magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);
    if (num == 10000) {
        cout << "success" << endl;
    }

    for (int i = 0; i < num; i++) {
        for (int x = 0; x < rows; x++) {
            for (int y = 0; y < cols; y++) {
                test_image[i].val[rows*x + y] = in(icin, 1);
            }
        }
    }
    icin.close();
    icin.open("data/t10k-labels.idx1-ubyte", std::ios::binary);
    magic = in(icin, 4), num = in(icin, 4);
    for (int i = 0; i < num; i++) {
        test_label[i] = in(icin, 1);
    }
    icin.close();

    load_network();

}

int main() {
    std::string base_dir = "/home/athul/dev/doone/data";
    //std::string base_dir = "/home/xy/caffe-master/data/mnist/";
    std::string img_path = base_dir + "train-images-idx3-ubyte";
    std::string label_path = base_dir + "train-labels-idx1-ubyte";

    input();

    Image im;
    int rows = 28; int cols = 28;
    for (int i = 0; i < 10; i++) {
        //cout << "next should be : " << label[i] << '\n';
        // im.num = label[i];
        // memcpy(im.val, image[i].val, n_input * sizeof(image[0].val[0]));
        // im.display();
        cout << label[i] << '\n';
        image[i].display();
    }
    test();

    // im.display();
    return 0;
}

