#include "sg-filter.hpp"

#include <iostream>
#include <math.h>

Eigen::VectorXd Filter::SgFilter::apply(Eigen::VectorXd const &a_x,
    uint32_t const a_windowLength, uint32_t const a_polyorder,
    uint32_t const a_deriv, double const a_delta, Mode const a_mode,
    double const a_constant)
{
  (void)a_constant;
  Eigen::VectorXd y;

  Eigen::VectorXd coeffs =
      getSavGolCoeffs(a_windowLength, a_polyorder, a_deriv, a_delta);

  switch (a_mode) {
  case Mode::Mirror:
  case Mode::Constant:
  case Mode::Nearest:
  case Mode::Wrap:
  default:
    // undefined for now
  case Mode::Interpolate:

    if (a_windowLength > a_x.size()) {
      std::cerr
          << "Filter: THe mode is 'interp' so a_windowLength must be less "
             "than or equal to the size of x."
          << std::endl;
    }
    y = Filter::convolve(a_x, coeffs);
    // Todo:Fix the borders
    break;
  }


  return y;
}

// std::vector<double> Filter::SgFilter::apply(std::vector<double> const
// &a_signal,
//     uint32_t const a_windowLength, uint32_t const a_polyorder,
//     uint32_t const a_deriv, double const a_delta, Mode const a_mode,
//     double const a_constant)
// {

//   // std::vector<double> results(a_array.size(),
//   //     convolve(Eigen::Map<double>(a_array.data(), a_array.size(), 1),
//   //         Eigen::Map<double>(a_kernel.data(), a_kernel.size(), 1))
//   //         .data());
//   return std::vector<double>();
// }

Eigen::VectorXd Filter::SgFilter::getSavGolCoeffs(uint32_t const a_windowLength,
    uint32_t const a_polyorder, uint32_t const a_deriv, double const a_delta,
    int32_t const a_pos, Method const a_use) noexcept
{
  Eigen::VectorXd coeffs;

  if (a_polyorder >= a_windowLength) {
    std::cerr << "sgfilter: a_polyorder must be less than a_windowLength."
              << std::endl;
  }

  // Windows bug with nan?
  int32_t pos = a_pos < 0 ? a_windowLength / 2 : a_pos;

  if (!(0 <= pos && pos < static_cast<int32_t>(a_windowLength))) {
    std::cerr
        << "sgfilter: pos must be nonnegative and less than window_length."
        << std::endl;
  }

  if (a_deriv > a_polyorder) {
    coeffs = Eigen::VectorXd(a_windowLength);
    coeffs.setZero();
  }

  Eigen::VectorXd x{Eigen::VectorXd::LinSpaced(
      a_windowLength, -pos, a_windowLength - pos - 1)};

  // std::cout << "x: " << x << std::endl;

  if (a_use == Method::Conv) {
    x.reverseInPlace();
  }

  Eigen::VectorXd order =
      Eigen::VectorXd::LinSpaced(a_polyorder + 1, 0, a_polyorder);

  // std::cout << "order: " << order << std::endl;

  Eigen::MatrixXd A = x.transpose()
                          .replicate(order.rows(), 1)
                          .array()
                          .pow(order.array().replicate(1, x.rows()));

  // std::cout << "A: " << A << std::endl;

  Eigen::VectorXd y(a_polyorder + 1);
  y.setZero();
  y[a_deriv] = std::tgamma(a_deriv + 1) / (std::pow(a_delta, a_deriv));
  // std::cout << "y: " << y << std::endl;
  // std::cout << "y: " << std::tgamma(a_deriv + 1) << std::endl;
  // std::cout << "y: " << std::pow(a_delta, a_deriv) << std::endl;

  // This should be in the future eigen api
  // coeffs =
  // A.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(
  //     y.transpose());
  // For now in 3.4.0... use this:
  coeffs = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
  // std::cout << "coeffs: " << coeffs.transpose() << std::endl;

  // A.bdcSvd()
  return coeffs;
}

template<class T>
Eigen::VectorX<T> Filter::convolve(
    Eigen::VectorX<T> const &a_array, Eigen::VectorX<T> const &a_kernel)
{

  Eigen::VectorX<T> results;

  if (a_array.size() < a_kernel.size()) {
    // std::clog << "Filter: convolution failed. kernel too big." << std::endl;
    return results;
  }
  results = a_array;
  uint64_t const offset = a_kernel.size() / 2;
  for (uint64_t n{offset}; n < a_array.size() - offset; n++) {
    results(n) = a_kernel.dot(
        a_array.segment(n - offset, a_kernel.size()).array().matrix());
  }


  // const Eigen::Index innersize = in.size() - (a_kernel.size()-1);
  // // Eigen::VectorXd out(in.size());
  // out.segment(offset, innersize) =
  //       in.head(innersize) * weights.x() +
  //       in.segment(1, innersize) * weights.y() +
  //       in.tail(innersize) * weights.z();
  return results;
}


template<class T>
std::vector<T> convolve(
    std::vector<T> const &a_array, std::vector<T> const &a_kernel)
{
  std::vector<T> results(a_array.size(),
      convolve(Eigen::Map<Eigen::VectorX<T>>(a_array.data(), a_array.size(), 1),
          Eigen::Map<Eigen::VectorX<T>>(a_kernel.data(), a_kernel.size(), 1))
          .data());
  return results;
}