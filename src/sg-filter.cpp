#include "sg-filter.hpp"

#include <iostream>
#include <math.h>

Eigen::VectorXd Filter::SgFilter::getSavGolCoeffs(uint32_t const a_windowLength, uint32_t const a_polyorder, uint32_t const a_deriv, double const a_delta, int32_t const a_pos, Method const a_use) noexcept
{
  Eigen::VectorXd coeffs;

  if (a_polyorder >= a_windowLength)
    std::cerr << "sgfilter: a_polyorder must be less than a_windowLength." << std::endl;

  // Windows bug with nan?
  int32_t pos = a_pos < 0 ? a_windowLength / 2 : a_pos;

  if (!(0 <= pos && pos < static_cast<int32_t>(a_windowLength)))
    std::cerr << "sgfilter: pos must be nonnegative and less than window_length." << std::endl;

  if (a_deriv > a_polyorder)
  {
    coeffs = Eigen::VectorXd(a_windowLength);
    coeffs.setZero();
  }

  Eigen::VectorXi x{Eigen::VectorXi::LinSpaced(a_windowLength, -pos, a_windowLength - pos)};

  std::cout << "x: " << x.transpose() << std::endl;

  if (a_use == Method::Conv)
  {
    coeffs.reverseInPlace();
  }

  Eigen::VectorXi order = x.LinSpaced(a_windowLength, 0, a_polyorder + 1);

  std::cout << "order: " << order.transpose() << std::endl;

  Eigen::VectorXi A = x.array().pow(order.array());
  std::cout << "A: " << A.transpose() << std::endl;

  Eigen::VectorXd y(a_polyorder + 1);
  y.setZero();
  y[a_deriv] = std::tgamma(a_deriv + 1) / (std::pow(a_delta, a_deriv));
  std::cout << "y: " << y.transpose() << std::endl;

  coeffs = A.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(y)

               return coeffs;
}