#pragma once

#include <Eigen/Dense>
#include <iostream>

namespace Filter {

class SgFilter {
 public:
  SgFilter() = default;
  virtual ~SgFilter() = default;

  enum Mode
  {
    Interpolate = 0,
    Mirror,
    Constant,
    Nearest,
    Wrap,
  };
  enum Method
  {
    Conv = 0,
    Dot
  };
  static Eigen::VectorXd apply(Eigen::VectorXd const &a_x,
      uint32_t const a_windowLength, uint32_t const a_polyorder,
      uint32_t const a_deriv = 0, double const a_delta = 1.0,
      Mode const a_mode = Mode::Interpolate, double const a_constant = 0.0);
  static std::vector<double> apply(std::vector<double> const &a_x,
      uint32_t const a_windowLength, uint32_t const a_polyorder,
      uint32_t const a_deriv = 0, double const a_delta = 1.0,
      Mode const a_mode = Mode::Interpolate, double const a_constant = 0.0);
  static Eigen::VectorXd getSavGolCoeffs(uint32_t const a_windowLength,
      uint32_t const a_polyorder, uint32_t const a_deriv = 0,
      double const a_delta = 1.0, int32_t a_pos = -1,
      Method const a_use = Method::Conv) noexcept;

 private:
};

template<class T>
Eigen::VectorX<T> convolve(
    Eigen::VectorX<T> const &a_array, Eigen::VectorX<T> const &a_kernel);
template<class T>
std::vector<T> convolve(
    std::vector<T> const &a_array, std::vector<T> const &a_kernel);
}
