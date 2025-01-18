#include <iostream>
#include <Eigen/Dense>

// Main function
int main() {
    
    Eigen::VectorXd sector_views(13);
    sector_views << 0.1467, 0.0, 0.0905, 0.0815, 0.0, 0.1192, 0.1238, 0.1676,
                    0.1156, 0.0, 0.1097, 0.0709, 0.0433;

    
    Eigen::VectorXd market_cap_weights(13);
    market_cap_weights << 0.0337, 0.0220, 0.0755, 0.1021, 0.0576, 0.1117, 0.1290,
                          0.3301, 0.0991, 0.0270, 0.0228, 0.1150, 0.1000;


    Eigen::VectorXd volatilities(13);
    volatilities << 0.3712, 0.2876, 0.2345, 0.2987, 0.1876, 0.2234, 0.2765,
                    0.3301, 0.2654, 0.1765, 0.2123, 0.3012, 0.0876;

    
    Eigen::MatrixXd correlation_matrix = Eigen::MatrixXd::Constant(13, 13, 0.3);
    for (int i = 0; i < 13; ++i) correlation_matrix(i, i) = 1.0;  // Diagonal = 1

    std::vector<std::pair<int, int>> higher_correlations = {
        {7, 8}, {3, 4}, {6, 10}  
    };
    for (auto& pair : higher_correlations) {
        correlation_matrix(pair.first, pair.second) = 0.5;
        correlation_matrix(pair.second, pair.first) = 0.5;
    }

    
    Eigen::MatrixXd cov_matrix(13, 13);
    for (int i = 0; i < 13; ++i) {
        for (int j = 0; j < 13; ++j) {
            cov_matrix(i, j) =
                correlation_matrix(i, j) * volatilities(i) * volatilities(j);
        }
    }

    // Risk aversion parameter
    double risk_aversion = 3.0;

    // Calculate equilibrium returns
    Eigen::VectorXd equilibrium_returns =
        risk_aversion * cov_matrix * market_cap_weights;

    // Views and confidence
    Eigen::VectorXd views = sector_views;  // Subjective expected returns
    Eigen::VectorXd confidence(13);
    confidence << 0.65, 0.0, 0.85, 0.75, 0.0, 0.38, 0.67, 0.55, 0.72, 0.0, 0.80, 0.88, 0.98;

    double confidenceeq = 0.5;  // Confidence in equilibrium returns

    // Combined returns using Black-Litterman formula
    Eigen::VectorXd combined_returns =
        (1 - confidenceeq) * equilibrium_returns + confidence.cwiseProduct(views);

    // Portfolio optimization (maximize Sharpe ratio)
    Eigen::MatrixXd A = cov_matrix;
    Eigen::VectorXd b = combined_returns;

    // Constraints: weights sum to 1
    Eigen::VectorXd weights = A.llt().solve(b);
    weights /= weights.sum();  // Normalize weights to sum to 1

    // Output optimized weights
    std::cout << "Optimized Weights:\n" << weights << "\n";

    // Calculate Sharpe ratio
    double portfolio_return = weights.dot(combined_returns);
    double portfolio_volatility =
        std::sqrt(weights.transpose() * cov_matrix * weights);
    double risk_free_rate = 0.0419;  // Annual bond yield
    double sharpe_ratio =
        (portfolio_return - risk_free_rate) / portfolio_volatility;

    std::cout << "Portfolio Sharpe Ratio: " << sharpe_ratio << "\n";

    return 0;
}
