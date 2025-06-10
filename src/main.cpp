#include "input.hpp"
#include "image.hpp"

using namespace std;
int main() {
    std::string base_dir = "/home/athul/dev/doone/data";
    //std::string base_dir = "/home/xy/caffe-master/data/mnist/";
    std::string img_path = base_dir + "train-images-idx3-ubyte";
    std::string label_path = base_dir + "train-labels-idx1-ubyte";

    input();

    Image im;
    rows = 28; cols = 28;
    std::string ascii = " .,:ilwW";
    for (int i = 0; i < 10; i++) {
        //cout << "next should be : " << label[i] << '\n';
        im.num = label[i];
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                int val = image[i][j][k];
                int pixel = 28*i + j;

                im.val[pixel] = val;
            }
        }
    }

    im.display();
    return 0;
}

