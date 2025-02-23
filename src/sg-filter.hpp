#pragma once

#include <Eigen/Dense>

namespace Filter
{

  class SgFilter
  {
  public:
    SgFilter() = default;
    virtual ~SgFilter() = default;

    enum Method
    {
      Conv = 0,
      Dot
    };
    static Eigen::VectorXd getSavGolCoeffs(uint32_t const a_windowLength, uint32_t const a_polyorder, uint32_t const a_deriv = 0, double const a_delta = 0.0, int32_t a_pos = -1, Method const a_use = Method::Conv) noexcept;

  private:
  };
}