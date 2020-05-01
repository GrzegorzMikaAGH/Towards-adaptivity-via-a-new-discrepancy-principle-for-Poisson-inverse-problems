// #include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
// #include <boost/multiprecision/float128.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/filesystem/fstream.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>



int main()
{
  // typedef boost::multiprecision::cpp_dec_float_50;
  using namespace boost::filesystem;
  // using namespace boost::lambda;
  using namespace boost::multiprecision;
  using boost::multiprecision::cpp_dec_float_50;
  unsigned int n_roots = 10000U;
  std::vector<cpp_dec_float_50> roots;
  boost::math::cyl_bessel_j_zero(-0.25, 1, n_roots, std::back_inserter(roots));
  
  path p{"bessel_zeros_short.txt"};
  ofstream ofs{p};
  ofs.precision(std::numeric_limits<cpp_dec_float_50>::digits10);
  std::copy(roots.begin(),
            roots.end(),
            std::ostream_iterator<cpp_dec_float_50>(ofs, "\n"));
} 
