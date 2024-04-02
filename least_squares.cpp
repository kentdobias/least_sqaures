#include <eigen3/Eigen/Dense>
#include <getopt.h>

#include "pcg-cpp/include/pcg_random.hpp"
#include "randutils/randutils.hpp"

#include "tensor.hpp"

Vector normalize(const Vector& z) {
  return z * sqrt((Real)z.size() / (Real)(z.transpose() * z));
}

template <int... ps>
class Model {
private:
  std::tuple<Tensor<ps>...> Js;

public:
  unsigned N;
  unsigned M;
  template <class Generator, typename... T>
  Model(unsigned N, unsigned M, Generator& r, T... μs) : N(N), M(M) {
    Js = std::make_tuple(μs * generateRealPSpinCouplings<ps>(N, M, r)...);
  }

  unsigned numPs() const {
    return std::tuple_size(Js);
  }

private:
  std::tuple<Vector, Matrix, Tensor<3>> hamGradTensorHelper(const Vector& z, const Tensor<1>& J) const {
    Tensor<3> J3(N, N, M);;
    J3.setZero();
    Matrix Jz = Matrix::Zero(N, M);
    Vector Jzz = Eigen::Map<const Vector>(J.data(), M);

    return {Jzz, Jz, J3};
  }

  std::tuple<Vector, Matrix, Tensor<3>> hamGradTensorHelper(const Vector& z, const Tensor<2>& J) const {
    Tensor<3> J3(N, N, M);;
    J3.setZero();
    Matrix Jz = Eigen::Map<const Matrix>(J.data(), N, M);
    Vector Jzz = z.transpose() * Jz;

    return {Jzz, Jz, J3};
  }

  template <int p>
  std::tuple<Vector, Matrix, Tensor<3>> hamGradTensorHelper(const Vector z, const Tensor<p>& J) const {
    Tensor<3> J3 = contractDown(J, z);
    Tensor<1> zT = Eigen::TensorMap<constTensor<1>>(z.data(), N);
    Tensor<2> J3zT = J3.contract(zT, ip00);
    Matrix Jz = Eigen::Map<const Matrix>(J3zT.data(), N, M);
    Vector Jzz = z.transpose() * Jz;

    return {Jzz, Jz, J3};
  }

  template <int p, int... qs>
  std::tuple<Vector, Matrix, Tensor<3>> hamGradHessHelper(const Vector& z, const Tensor<p>& J, const Tensor<qs>& ...Js) const {
    auto [Jzz, Jz, J3] = hamGradTensorHelper(z, J);

    Real pBang = factorial(p-1);

    Tensor<3> ddH = ((p - 1) * p / pBang) * J3;
    Matrix dH = (p / pBang) * Jz;
    Vector H = Jzz / pBang;

    if constexpr (sizeof...(Js) > 0) {
      auto [Hs, dHs, ddHs] = hamGradHessHelper(z, Js...);
      H += Hs;
      dH += dHs;
      ddH += ddHs;
    }

    return {H, dH, ddH};
  }

public:
  std::tuple<Vector, Matrix, Tensor<3>> VdVddV(const Vector& z) const {
    return std::apply([&z, this](const Tensor<ps>& ...Ks) -> std::tuple<Vector, Matrix, Tensor<3>> { return hamGradHessHelper(z, Ks...); }, Js);
  }

  std::tuple<Real, Vector, Matrix> HdHddH(const Vector& z) const {
    auto [V, dV, ddV] = VdVddV(z);

    Real H = 0.5 * V.squaredNorm();
    Vector dH = dV * V;
    Tensor<1> VT = Eigen::TensorMap<constTensor<1>>(V.data(), M);
    Tensor<2> ddVzT = ddV.contract(VT, ip20);
    Matrix ddH = Eigen::Map<const Matrix>(ddVzT.data(), N, N) + dV * dV.transpose();

    return {H, dH, ddH};
  }

  std::tuple<Real, Vector, Matrix> hamGradHess(const Vector& x) const {
    auto [H, dH, ddH] = HdHddH(x);

    Vector gradH = dH - dH.dot(x) * x / (Real)N;
    Matrix hessH = ddH - (dH * x.transpose() + x.dot(dH) * Matrix::Identity(N, N) + (ddH * x) * x.transpose()) / (Real)N  + 2.0 * x * x.transpose();

    return {H, gradH, hessH};
  }

  Vector HessSpectrum(const Vector& x) const {
    Matrix hessH;
    std::tie(std::ignore, std::ignore, hessH) = hamGradHess(x);
    Eigen::EigenSolver<Matrix> eigenS(hessH);
    return eigenS.eigenvalues().real();
  }
};

template <int ...ps>
Vector findMinimum(const Model<ps...>& M, const Vector& x0, Real ε) {
  Vector x = x0;
  Real λ = 100;

  auto [H, g, m] = M.hamGradHess(x0);

  while (g.norm() / x.size() > ε && λ < 1e8) {
    Vector dz = (m + λ * (Matrix)abs(m.diagonal().array()).matrix().asDiagonal()).partialPivLu().solve(g);
    dz -= x.dot(dz) * x / M.N;
    Vector zNew = normalize(x - dz);

    auto [HNew, gNew, mNew] = M.hamGradHess(zNew);

    if (HNew * 1.0001 <= H) {
      x = zNew;
      H = HNew;

      g = gNew;
      m = mNew;

      λ /= 2;
    } else {
      λ *= 1.5;
    }
  }

  return x;
}

using Rng = randutils::random_generator<pcg32>;
using Real = double;

int main(int argc, char* argv[]) {
  unsigned N = 10;
  Real α = 1;
  Real σ = 1;

  int opt;

  while ((opt = getopt(argc, argv, "N:a:s:")) != -1) {
    switch (opt) {
    case 'N':
      N = (unsigned)atof(optarg);
      break;
    case 'a':
      α = atof(optarg);
      break;
    case 's':
      σ = atof(optarg);
      break;
    default:
      exit(1);
    }
  }

  unsigned M = (unsigned)(α * N);

  Rng r;

  Model<1, 2> leastSquares(N, M, r.engine(), σ, 1);

  Vector x = Vector::Zero(N);
  x(0) = sqrt(N);

  double energy;
  std::tie(energy, std::ignore, std::ignore) = leastSquares.hamGradHess(x);

  std::cout << energy / N << std::endl;

  Vector xMin = findMinimum(leastSquares, x, 1e-12);
  std::tie(energy, std::ignore, std::ignore) = leastSquares.hamGradHess(xMin);

  std::cout << energy / N << std::endl;
  std::cout << leastSquares.HessSpectrum(xMin)(1) / N << std::endl;

  return 0;
}
