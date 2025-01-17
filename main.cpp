#include <iostream>
#include <Eigen/Dense>  // Include the Eigen library

int main() {
    // Define a 2x2 matrix using Eigen
    Eigen::MatrixXd m(2, 2);

    // Set the matrix elements
    m(0, 0) = 3;            // First row, first column
    m(1, 0) = 2.5;          // Second row, first column
    m(0, 1) = -1;           // First row, second column
    m(1, 1) = m(1, 0) + m(0, 1);  // Second row, second column = 2.5 + (-1)

    // Print the matrix
    std::cout << "Matrix:\n" << m << std::endl;

    return 0;
}