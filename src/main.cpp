#include <iostream>
#include <string>
#include <fstream>
#include <stdint.h>
#include <vector>
#include <cmath>

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
const int n_layers = (int)n_neurons.size();

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

vector<vector<vector<double>>> w(n_layers - 1); // Weights: w[0][3][4] means this is weights from 0th layer to 1st layer.
                                                // it is from index 3 from layer 0 and index 4 from layer 4
vector<vector<double>> b(n_layers + 1); // Bias: b[3][2] means the bias of neuron i layer 3 and index 2
                                        // b[0] is initialised with 0's
vector<vector<double>> a(n_layers); // Current value of each node

// VARIABLES END

unsigned int in(std::ifstream& icin, unsigned int inp_size) {
    unsigned int ans = 0;
    for (int i = 0; i < inp_size; i++) {
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

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// If a trainined network is not found, start from beginning
// Initialising weights and biases to random values.
void initialize_network() {
    for (int i = 0; i < n_layers; i++) {
        a[i].resize(n_neurons[i]);
        b[i].resize(n_neurons[i]);
        if (i < n_layers - 1) {
            w[i].resize(n_neurons[i], vector<double>(n_neurons[i+1]));
        }
    }
    
    for (int l = 0; l < n_layers - 1; l++) {
        for (int i = 0; i < n_neurons[l]; i++) {
            for (int j = 0; j < n_neurons[l + 1]; j++) {
                int sign = rand() % 2;
                w[l][i][j] = (double)(rand() % 6) / 10.0;
                if (sign) {
                    w[l][i][j] = -w[l][i][j];
                }
            }
        }
    }
    for (int i = 0; i < n_neurons[0]; i++) {
        b[0][i] = (double)0.0;
    }
    for (int l = 1; l < n_layers; l++) {
        for (int i = 0; i < n_neurons[l]; i++) {
            int sign = rand() % 2;
            b[l][i] = (double)(rand() % 6) / 10.0;
            if (sign) {
                b[l][i] = -b[l][i];
            }
        }
    }
}

// if there is already a saved file, then load weights & biases from there
// otherwise, initialize those randomly.
void load_network() {
    string net_filename = "network/";
    for (int i = 0; i < n_neurons.size(); i++) {
        if (i) { net_filename += '_'; }
        net_filename += to_string(n_neurons[i]);
    }

    initialize_network();

    ifstream saved_net(net_filename);
    if (!saved_net.fail()) {
        cout << "No trained files found.\nStarting from random weights and biases" << endl;
        for (int l = 0; l < n_layers - 1; l++) {
            for (int i = 0; i < n_neurons[l]; i++) {
                for (int j = 0; j < n_neurons[l + 1]; j++) {
                    saved_net >> w[l][i][j];
                }
            }
        }
        for (int l = 0; l < n_layers; l++) {
            for (int i = 0; i < n_neurons[l]; i++) {
                saved_net >> b[l][i];
            }
        }
    }
}

// save the current weights and biases into a file
void write_network() {
    string net_filename = "network/";
    for (int i = 0; i < n_neurons.size(); i++) {
        if (i) { net_filename += '_'; }
        net_filename += to_string(n_neurons[i]);
    }

    ofstream fout(net_filename);
    for (int l = 0; l < n_layers - 1; l++) {
        for (int i = 0; i < n_neurons[l]; i++) {
            for (int j = 0; j < n_neurons[l + 1]; j++) {
                fout << w[l][i][j] << " ";
            }
        }
    }
    for (int l = 0; l < n_layers; l++) {
        for (int i = 0; i < n_neurons[l]; i++) {
            fout << b[l][i] << " ";
        }
    }
    fout.close();
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
    //test();
    initialize_network();
    write_network();

    // im.display();
    return 0;
}

