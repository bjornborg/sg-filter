#include "sg-filter.hpp"
#include <Eigen/Dense>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <iostream>

TEST_CASE("Sg filter - Coeffs")
{
  {
    Eigen::VectorXd coeffs = Filter::SgFilter::getSavGolCoeffs(5, 2);
    REQUIRE(coeffs.isApprox((Eigen::VectorXd(5) << -0.08571429, 0.34285714,
                                0.48571429, 0.34285714, -0.08571429)
                                .finished(),
        1e-4));
  }
  {
    Eigen::VectorXd coeffs = Filter::SgFilter::getSavGolCoeffs(5, 2, 1);
    REQUIRE(
        coeffs.isApprox((Eigen::VectorXd(5) << 2.00000000e-01, 1.00000000e-01,
                            2.07548111e-16, -1.00000000e-01, -2.00000000e-01)
                            .finished(),
            1e-4));
  }
  {
    Eigen::VectorXd coeffs = Filter::SgFilter::getSavGolCoeffs(5, 2, 0, 1.0, 3);
    REQUIRE(coeffs.isApprox((Eigen::VectorXd(5) << 0.25714286, 0.37142857,
                                0.34285714, 0.17142857, -0.14285714)
                                .finished(),
        1e-4));
  }
  {
    Eigen::VectorXd coeffs = Filter::SgFilter::getSavGolCoeffs(
        5, 2, 0, 1.0, 3, Filter::SgFilter::Dot);
    REQUIRE(coeffs.isApprox((Eigen::VectorXd(5) << -0.14285714, 0.17142857,
                                0.34285714, 0.37142857, 0.25714286)
                                .finished(),
        1e-4));
  }
  {
    Eigen::VectorXd coeffs = Filter::SgFilter::getSavGolCoeffs(
        4, 2, 1, 1.0, 3, Filter::SgFilter::Dot);
    REQUIRE(coeffs.isApprox(
        (Eigen::VectorXd(4) << 0.45, -0.85, -0.65, 1.05).finished(), 1e-4));
  }
  {
    Eigen::VectorXd coeffs = Filter::SgFilter::getSavGolCoeffs(
        5, 2, 1, 1.0, 4, Filter::SgFilter::Dot);
    Eigen::VectorXd x(5);
    x << 1, 0, 1, 4, 9;
    REQUIRE_THAT(coeffs.dot(x), Catch::Matchers::WithinRel(6.0));
  }
}


TEST_CASE("Sg filter - Apply")
{
  // std vector
  {
    std::vector<double> x{2, 2, 5, 2, 1, 0, 1, 4, 9};
    std::vector<double> yref{
        2, 2, 3.54286, 2.85714, 0.657143, 0.171429, 1, 4, 9};
    std::vector<double> y = Filter::SgFilter::apply(x, 5, 3);
    REQUIRE_THAT(y, Catch::Matchers::Approx(yref));
  }
  // Eigen
  {
    Eigen::VectorXd x(9);
    x << 2, 2, 5, 2, 1, 0, 1, 4, 9;
    Eigen::VectorXd yref = x;
    yref << 2, 2, 3.54286, 2.85714, 0.657143, 0.171429, 1, 4, 9;
    Eigen::VectorXd y = Filter::SgFilter::apply(x, 5, 3);
    // std::cout << "sgfilter y: " << y.transpose() << std::endl;
    REQUIRE(yref.isApprox(y, 1e-4));
  }
}