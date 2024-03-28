#include <eigen3/Eigen/Dense>
#include <random>
#include <getopt.h>

#include "pcg-cpp/include/pcg_random.hpp"
#include "randutils/randutils.hpp"


template <class Real>
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

template <class Real>
using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

template <class Real>
class Model {
private:
  Matrix<Real> A;
  Vector<Real> b;

public:
  template <class Generator>
  Model(Real σ, unsigned N, unsigned M, Generator& r) : A(M, N), b(M) {
    std::normal_distribution<Real> aDistribution(0, 1 / sqrt(N));

    for (unsigned i = 0; i < M; i++) {
      for (unsigned j =0; j < N; j++) {
        A(i, j) = aDistribution(r);
      }
    }

    std::normal_distribution<Real> bDistribution(0, σ);

    for (unsigned i = 0; i < M; i++) {
      b(i) = bDistribution(r);
    }
  }

  unsigned N() const {
    return A.cols();
  }

  unsigned M() const {
    return A.rows();
  }

  Vector<Real> V(const Vector<Real>& x) const {
    return A * x + b;
  }

  Matrix<Real> dV(const Vector<Real>& x) const {
    return A;
  }

//  const Real ddV(const Vector<Real>& x) {
//    return Matrix::Zero(;
//  }

  Real H(const Vector<Real>& x) const {
    return V(x).squaredNorm();
  }

  Vector<Real> dH(const Vector<Real>& x) const {
    return dV(x).transpose() * V(x);
  }

  Matrix<Real> ddH(const Vector<Real>& x) const {
    return dV(x).transpose() * dV(x);
  }

  Vector<Real> ∇H(const Vector<Real>& x) const {
    return dH(x) - dH(x).dot(x) * x / x.squaredNorm();
  }

  Matrix<Real> HessH(const Vector<Real>& x) const {
    Matrix<Real> hess = ddH(x) - x.dot(dH(x)) * Matrix<Real>::Identity(N(), N());
    return hess - (hess * x) * x.transpose() / x.squaredNorm();
  }

  Vector<Real> HessSpectrum(const Vector<Real>& x) const {
    Eigen::EigenSolver<Matrix<Real>> eigenS(HessH(x));
    return eigenS.eigenvalues().real();
  }
};

template <typename Derived>
Vector<typename Derived::Scalar> normalize(const Eigen::MatrixBase<Derived>& z) {
  return z * sqrt((double)z.size() / (typename Derived::Scalar)(z.transpose() * z));
}

template <class Real>
Vector<Real> findMinimum(const Model<Real>& M, const Vector<Real>& x0, Real ε) {
  Vector<Real> x = x0;
  Real λ = 100;

  Real H = M.H(x);
  Vector<Real> dH = M.dH(x);
  Matrix<Real> ddH = M.ddH(x);

  Vector<Real> g = dH - x.dot(dH) * x / x.squaredNorm();
  Matrix<Real> m = ddH - (dH * x.transpose() + x.dot(dH) * Matrix<Real>::Identity(M.N(), M.N()) + (ddH * x) * x.transpose()) / x.squaredNorm() + 2.0 * x * x.transpose();

  while (g.norm() / x.size() > ε && λ < 1e8) {
    Vector<Real> dz = (m + λ * (Matrix<Real>)abs(m.diagonal().array()).matrix().asDiagonal()).partialPivLu().solve(g);
    dz -= x.dot(dz) * x / x.squaredNorm();
    Vector<Real> zNew = normalize(x - dz);

    Real HNew = M.H(zNew);
    Vector<Real> dHNew = M.dH(zNew);
    Matrix<Real> ddHNew = M.ddH(zNew);

    if (HNew * 1.0001 <= H) {
      x = zNew;
      H = HNew;
      dH = dHNew;
      ddH = ddHNew;

      g = dH - x.dot(dH) * x / (Real)x.size();
      m = ddH - (dH * x.transpose() + x.dot(dH) * Matrix<Real>::Identity(x.size(), x.size()) + (ddH * x) * x.transpose()) / (Real)x.size() + 2.0 * x * x.transpose();

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

  Model<Real> leastSquares(σ, N, M, r.engine());

  Vector<Real> x = Vector<Real>::Zero(N);
  x(0) = sqrt(N);

  std::cout << leastSquares.H(x) / N << std::endl;

  Vector<Real> xMin = findMinimum(leastSquares, x, 1e-12);

  std::cout << leastSquares.H(xMin) / N << std::endl;
  std::cout << leastSquares.HessSpectrum(xMin)(1) / N << std::endl;

  return 0;
}
