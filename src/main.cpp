#include <iostream>
#include <string>
#include <fstream>
#include <stdint.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>

using namespace std;

const int width = 28;
const int height = 28;

const int n_input = width * height; // Number of input neurons (784 pixels of an image)
const int n_train_images = 60000; // No. of training images
const int n_test_images = 10000; // No. of images for testing

// no. of neurons in each layers.
const vector<int> n_neurons = {n_input, 10, 10};
const int n_layers = (int)n_neurons.size();

// epochs = Number of iterations of entire dataset while training
// learning_rate = Learing rate
const int epochs = 20;
const double learning_rate = 0.1;

struct Image{
    private:

    public:
        double val[784];

        void display() {
            char ascii[] = " .,:ilwW";

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int pixel = val[width*i + j];
                    std::cout << ascii[pixel/32] << " ";
                }
                std::cout << std::endl;
            }
        }
};

Image image[n_train_images];
unsigned int label[n_train_images]; // Correct label of the training images

Image test_image[n_test_images];
unsigned int test_label[n_test_images]; // the correct label of the test images

vector<vector<vector<double>>> w(n_layers - 1); // Weights: w[0][3][4] means this is weights from 0th layer to 1st layer. it is from index 3 from layer 0 and index 4 from layer 4
vector<vector<vector<double>>> cum_delta_w(n_layers - 1); // Sum of error in w in each layer over each mini-batch images
vector<vector<double>> b(n_layers + 1); // Bias: b[3][2] means the bias of neuron i layer 3 and index 2. b[0] is initialised with 0's
vector<vector<double>> cum_delta_b(n_layers + 1); // 
vector<vector<double>> a(n_layers); // Current value of each node
vector<vector<double>> z(n_layers); // non-sigmoided value of each neuron
vector<vector<double>> delta(n_layers); // Current errors in each layer for each image

// Take input from binary file 
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

void load_images() {
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

// Derivative of sigmoid function
double sigmoid_prime(double x) {
    return sigmoid(x)*(1-sigmoid(x));
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
        cout << "File: '" << net_filename << "' found. \nStarting from pre-trained weights & biases." << endl;
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
    } else {
        cout << "File: '" << net_filename << "' not found. \bStarting from random weights & biases." << endl;
    }
}

// Save the current weights and biases into a file
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

// Reset the values of a and delta after each image
void reset_variables() {
    for (int l = 0; l < n_layers; l++) {
        a[l].resize(n_neurons[l]);
        z[l].resize(n_neurons[l]);
        delta[l].resize(n_neurons[l]);
        for (int i = 0; i < n_neurons[l]; i++) {
            a[l][i] = 0;
            z[l][i] = 0;
            delta[l][i] = 0;
        }
    }
}

void feed_forward() {
    for (int l = 1; l < n_layers; l++) {
        for (int i = 0; i < n_neurons[l]; i++) {
            double v = 0.0;
            for (int j = 0; j < n_neurons[l - 1]; j++) {
                v += a[l - 1][j]*w[l - 1][j][i];
            }
            v += b[l][i];
            z[l][i] = v;
            a[l][i] = sigmoid(v);
        }
    }
}

// Find the maximum of the last layer to find the guess
int guess() {
    int g = 0;
    double mx = 0.0;
    for (int i = 0; i < n_neurons[n_layers - 1]; i++) {
        if (a[n_layers - 1][i] > mx) {
            mx = a[n_layers - 1][i];
            g = i;
        }
    }
    return g;
}

void train() {
    int n_batch = 1000; // No. of training samples in each mini-batch
    vector<int> id(n_train_images);
    std::iota(id.begin(), id.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(id.begin(), id.end(), gen);

    // initialise cum_delta_w
    for (int l = 0; l < n_layers - 1; l++) {
        cum_delta_w[l].resize(n_neurons[l], vector<double>(n_neurons[l+1], 0.0));
    }
    for (int l = 0; l < n_layers; l++) {
        cum_delta_b[l].resize(n_neurons[l], 0.0);
    }

    for (int tt = 0; tt < epochs; tt++) {
        int correct = 0;
        for (int num = 0; num < n_train_images; num++) {
            reset_variables();
            // Copy the new image to input layer
            // Dividing by 254 to make a[0] values in range (0,1)
            for (int i = 0; i < n_input; i++) {
                a[0][i] = image[id[num]].val[i]/254;
            }
            feed_forward();

            int g = guess();
            if (label[id[num]] == g) {
                correct++;
            }

            // error in final layer
            for (int j = 0; j < n_neurons[n_layers - 1]; j++) {
                double v = (label[id[num]] == j? 1.0 : 0.0);
                delta[n_layers - 1][j] = 2*(a[n_layers - 1][j] - v)*sigmoid_prime(z[n_layers - 1][j]);
            }
            
            for (int l = n_layers - 2; l >= 1; l--) {
                for (int j = 0; j < n_neurons[l]; j++) {
                    for (int k = 0; k < n_neurons[l + 1]; k++) {
                        delta[l][j] += w[l][j][k]*delta[l + 1][k]*sigmoid_prime(z[l][j]);
                    }
                }
            }

            for (int l = 0; l < n_layers - 1; l++) {
                for (int i = 0; i < n_neurons[l]; i++) {
                    for (int j = 0; j < n_neurons[l + 1]; j++) {
                        cum_delta_w[l][i][j] += a[l][i]*delta[l + 1][j];
                    }
                    cum_delta_b[l][i] += delta[l][i];
                }
            }

            // If a set of n_batch images are completed
            if (num % n_batch == (n_batch - 1)) {
                // Modify weights
                for (int l = 0; l < n_layers - 1; l++) {
                    for (int k = 0; k < n_neurons[l]; k++) {
                        for (int j = 0; j < n_neurons[l + 1]; j++) {
                            w[l][k][j] -= (learning_rate*cum_delta_w[l][k][j])/n_batch;
                        }
                    }
                }
                // Modify bias
                for (int l = 1; l < n_layers; l++) {
                    for (int i = 0; i < n_neurons[l]; i++) {
                        b[l][i] -= (learning_rate*cum_delta_b[l][i])/n_batch;
                    }
                }

                cout << setprecision(2) << fixed;
                cout << "Completion: " << (double)(tt*n_train_images + num + 1)*100.0/(double)(epochs*n_train_images) << "%  ";
                cout << (double)(correct * 100.0)/n_batch << "% accuracy" << endl;
                correct = 0;

                // Reset cum_delta_w and cum_delta_b after each batch
                for (int l = 0; l < n_layers; l++) {
                    for (int i = 0; i < n_neurons[l]; i++) {
                        for (int j = 0; j < n_neurons[l + 1]; j++) {
                            cum_delta_w[l][i][j] = 0.0;
                        }
                    }
                }
                for (int l = 0; l <= n_layers; l++) {
                    for (int i = 0; i < n_neurons[l]; i++) {
                        cum_delta_b[l][i] = 0.0;
                    }
                }

                write_network();
            }
        }
    }
}

void test() {
    std::ifstream icin;
    icin.open("data/t10k-images.idx3-ubyte", std::ios::binary);

    unsigned int magic, num, rows, cols; // Metadata. won't be using
    magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);

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

    int correct = 0;
    for (int n = 0; n < n_test_images; n++) {
        reset_variables();
        for (int i = 0; i < 784; i++) {
            a[0][i] = test_image[n].val[i]/254;
        }
        feed_forward();
        if (guess() == test_label[n]) {
            correct++;
        }
    }
    cout << (double)(correct * 100.0)/n_test_images << "% accuracy" << endl;
}

int main() {
    load_images();
    reset_variables();

    //train();
    test();
    return 0;
}

