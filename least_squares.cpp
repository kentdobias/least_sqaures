#include <eigen3/Eigen/Dense>
#include <getopt.h>

#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#include "pcg-cpp/include/pcg_random.hpp"
#include "randutils/randutils.hpp"

using Rng = randutils::random_generator<pcg32>;

using Real = double;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

/* Eigen tensor manipulations are quite annoying, especially with the need to convert other types
 * into tensors beforehand. Here I overload multiplication operators to allow contraction between
 * vectors and the first or last index of a tensor.
 */
class Tensor : public Eigen::Tensor<Real, 3> {
  using Eigen::Tensor<Real, 3>::Tensor;

public:
  Matrix operator*(const Vector& x) const {
    const std::array<Eigen::IndexPair<int>, 1> ip20 = {Eigen::IndexPair<int>(2, 0)};
    const Eigen::Tensor<Real, 1> xT = Eigen::TensorMap<const Eigen::Tensor<Real, 1>>(x.data(), x.size());
    const Eigen::Tensor<Real, 2> JxT = contract(xT, ip20);
    return Eigen::Map<const Matrix>(JxT.data(), dimension(0), dimension(1));
  }
};

Matrix operator*(const Eigen::Matrix<Real, 1, Eigen::Dynamic>& x, const Tensor& J) {
  const std::array<Eigen::IndexPair<int>, 1> ip00 = {Eigen::IndexPair<int>(0, 0)};
  const Eigen::Tensor<Real, 1> xT = Eigen::TensorMap<const Eigen::Tensor<Real, 1>>(x.data(), x.size());
  const Eigen::Tensor<Real, 2> JxT = J.contract(xT, ip00);
  return Eigen::Map<const Matrix>(JxT.data(), J.dimension(1), J.dimension(2));
}

Vector normalize(const Vector& x) {
  return x * sqrt((Real)x.size() / x.squaredNorm());
}

class QuadraticModel {
private:
  Tensor J;
  Matrix A;
  Vector b;

public:
  unsigned N;
  unsigned M;

  template <class Generator>
  QuadraticModel(unsigned N, unsigned M, Generator& r, double μ1, double μ2, double μ3) : N(N), M(M), J(M, N, N), A(M, N), b(M) {
    std::normal_distribution<Real> distribution(0, 1);

    for (unsigned i = 0; i < M; i++) {
      for (unsigned j = 0; j < N; j++) {
        for (unsigned k = 0; k < N; k++) {
          J(i, j, k) = (2 * μ3 / N) * distribution(r);
        }
      }
    }

    for (unsigned i = 0; i < M; i++) {
      for (unsigned j = 0; j < N; j++) {
        A(i, j) = (μ2 / sqrt(N)) * distribution(r);
      }
    }

    for (unsigned i = 0; i < M; i++) {
      b(i) = μ1 * distribution(r);
    }
  }

  std::tuple<Vector, Matrix, const Tensor&> VdVddV(const Vector& x) const {
    Matrix Jx = J * x;
    Vector V1 = (A + 0.5 * Jx) * x;
    Matrix dV = A + Jx;

    return {b + V1, dV, J};
  }

  std::tuple<Real, Vector, Matrix> HdHddH(const Vector& x) const {
    auto [V, dV, ddV] = VdVddV(x);

    Real H = 0.5 * V.squaredNorm();
    Vector dH = V.transpose() * dV;
    Matrix ddH = V.transpose() * ddV + dV.transpose() * dV;

    return {H, dH, ddH};
  }

  std::tuple<Real, Vector, Matrix> hamGradHess(const Vector& x) const {
    auto [H, dH, ddH] = HdHddH(x);

    Vector gradH = dH - dH.dot(x) * x / (Real)N;
    Matrix hessH = ddH - (dH * x.transpose() + x.dot(dH) * Matrix::Identity(N, N) + (ddH * x) * x.transpose()) / (Real)N  + 2.0 * x * x.transpose();

    return {H, gradH, hessH};
  }

  Vector spectrum(const Vector& x) const {
    Matrix hessH;
    std::tie(std::ignore, std::ignore, hessH) = hamGradHess(x);
    Eigen::EigenSolver<Matrix> eigenS(hessH);
    return eigenS.eigenvalues().real();
  }
};

Vector findMinimum(const QuadraticModel& M, const Vector& x0, Real ε) {
  Vector x = x0;
  Real λ = 100;

  auto [H, g, m] = M.hamGradHess(x0);

  while (g.norm() / x.size() > ε && λ < 1e8) {
    Vector dx = (m + λ * (Matrix)abs(m.diagonal().array()).matrix().asDiagonal()).partialPivLu().solve(g);
    dx -= x.dot(dx) * x / M.N;
    Vector xNew = normalize(x - dx);

    auto [HNew, gNew, mNew] = M.hamGradHess(xNew);

    if (HNew * 1.0001 <= H) {
      x = xNew;
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

int main(int argc, char* argv[]) {
  unsigned N = 10;
  Real α = 1;
  Real σ = 1;
  Real A = 1;
  Real J = 1;

  int opt;

  while ((opt = getopt(argc, argv, "N:a:s:A:J:")) != -1) {
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
    case 'A':
      A = atof(optarg);
      break;
    case 'J':
      J = atof(optarg);
      break;
    default:
      exit(1);
    }
  }

  unsigned M = (unsigned)(α * N);

  Rng r;

  QuadraticModel leastSquares(N, M, r.engine(), σ, A, J);

  Vector x = Vector::Zero(N);
  x(0) = sqrt(N);

  double energy;
  std::tie(energy, std::ignore, std::ignore) = leastSquares.hamGradHess(x);

  std::cout << energy / N << std::endl;

  Vector xMin = findMinimum(leastSquares, x, 1e-12);
  std::tie(energy, std::ignore, std::ignore) = leastSquares.hamGradHess(xMin);

  std::cout << energy / N << std::endl;
  std::cout << leastSquares.spectrum(xMin)(1) / N << std::endl;

  return 0;
}
