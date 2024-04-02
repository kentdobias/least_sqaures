#pragma once

#include <array>
#include <functional>

#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#include "types.hpp"
#include "factorial.hpp"

using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

template <int p>
using Tensor = Eigen::Tensor<Real, p>;

template <int p>
using constTensor = Eigen::Tensor<const Real, p>;

template <int p, std::size_t... Indices>
Tensor<p> initializeJHelper(unsigned N, unsigned M, std::index_sequence<Indices...>) {
  std::array<unsigned, p> Ns;
  std::fill_n(Ns.begin(), p, N);
  Ns[p-1] = M;
  return Tensor<p>(std::get<Indices>(Ns)...);
}

template <int p>
Tensor<p> initializeJ(unsigned N, unsigned M) {
  return initializeJHelper<p>(N, M, std::make_index_sequence<p>());
}

template <int p, std::size_t... Indices>
void setJHelper(Tensor<p>& J, const std::array<unsigned, p>& ind, Real val, std::index_sequence<Indices...>) {
  J(std::get<Indices>(ind)...) = val;
}

template <int p>
void setJ(Tensor<p>& J, std::array<unsigned, p> ind, Real val) {
  setJHelper<p>(J, ind, val, std::make_index_sequence<p>());
}

template <int p, std::size_t... Indices>
Real getJHelper(const Tensor<p>& J, const std::array<unsigned, p>& ind, std::index_sequence<Indices...>) {
  return J(std::get<Indices>(ind)...);
}

template <int p>
Real getJ(const Tensor<p>& J, const std::array<unsigned, p>& ind) {
  return getJHelper<p>(J, ind, std::make_index_sequence<p>());
}

template <int p>
void iterateOverHelper(Tensor<p>& J,
    std::function<void(Tensor<p>&, std::array<unsigned, p>)>& f,
    unsigned l, std::array<unsigned, p> is) {
  if (l == 0) {
    f(J, is);
  } else {
    for (unsigned i = 0; i < J.dimension(p - l); i++) {
      std::array<unsigned, p> js = is;
      js[p - l] = i;
      iterateOverHelper<p>(J, f, l - 1, js);
    }
  }
}

template <int p>
void iterateOver(Tensor<p>& J, std::function<void(Tensor<p>&, std::array<unsigned, p>)>& f) {
  std::array<unsigned, p> is;
  iterateOverHelper<p>(J, f, p, is);
}

template <int p, class Distribution, class Generator>
Tensor<p> generateCouplings(unsigned N, unsigned M, Distribution d, Generator& r) {
  Tensor<p> J = initializeJ<p>(N, M);

  std::function<void(Tensor<p>&, std::array<unsigned, p>)> setRandom =
    [&d, &r] (Tensor<p>& JJ, std::array<unsigned, p> ind) {
      setJ<p>(JJ, ind, d(r));
    };

  iterateOver<p>(J, setRandom);

  return J;
}

template <int p, class Generator>
Tensor<p> generateRealPSpinCouplings(unsigned N, unsigned M, Generator& r) {
  Real σp = sqrt(factorial(p-1) / ((Real)2 * pow(N, p - 1)));

  return generateCouplings<p>(N, M, std::normal_distribution<Real>(0, σp), r);
}

Tensor<3> contractDown(const Tensor<3>& J, const Vector& z) {
  return J;
}

const std::array<Eigen::IndexPair<int>, 1> ip00 = {Eigen::IndexPair<int>(0, 0)};
const std::array<Eigen::IndexPair<int>, 1> ip20 = {Eigen::IndexPair<int>(2, 0)};

template <int r>
Tensor<3> contractDown(const Tensor<r>& J, const Vector& z) {
  Tensor<1> zT = Eigen::TensorMap<constTensor<1>>(z.data(), {z.size()});
  Tensor<r - 1> Jz = J.contract(zT, ip00);
  return contractDown(Jz, z);
}
