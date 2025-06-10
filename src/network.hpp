#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include "image.hpp"

struct Network{
    private:
        const int MAXN = 6e4;
        std::vector<Image> data_image(MAXN);
        std::vector<unsigned int> data_label(MAXN);

    public:
};

#endif // NETWORK_HPP
