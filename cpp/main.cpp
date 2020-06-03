#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(3);
    std::cout << v << std::endl;
    std::cout << "Hello, world!" << std::endl;
    return 0;
}
