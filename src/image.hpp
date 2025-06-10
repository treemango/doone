#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <iostream>

struct Image{
    private:

    public:
        unsigned int num;
        unsigned int val[784];
        void display() {
            std::cout << "Next number should be : " << num << '\n';
            char ascii[] = " .,:ilwW";

            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    int pixel = val[28*i + j];
                    std::cout << pixel << '\n';
                    std::cout << ascii[(val[pixel])/64];
                }
                std::cout << std::endl;
            }
        }
};

#endif // IMAGE_HPP

