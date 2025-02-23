#include "sg-filter.hpp"
#include <Eigen/Dense>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Sg filter")
{
  Filter::SgFilter::getSavGolCoeffs(3, 1);
  REQUIRE(true);
}