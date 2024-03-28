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
    std::normal_distribution<Real> aDistribution(0, 1);

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

  const unsigned N() {
    return A.cols();
  }

  const unsigned M() {
    return A.rows();
  }

  const Vector<Real> V(const Vector<Real>& x) {
    return A * x + b;
  }

  const Matrix<Real> dV(const Vector<Real>& x) {
    return A;
  }

//  const Real ddV(const Vector<Real>& x) {
//    return Matrix::Zero(;
//  }

  const Real H(const Vector<Real>& x) {
    return V(x).squaredNorm();
  }

  const Vector<Real> dH(const Vector<Real>& x) {
    return dV(x).transpose() * V(x);
  }

  const Matrix<Real> ddH(const Vector<Real>& x) {
    return dV(x).transpose() * dV(x);
  }

  const Vector<Real> ∇H(const Vector<Real>& x){
    return dH(x) - dH(x).dot(x) * x / x.squaredNorm();
  }

  const Matrix<Real> HessH(const Vector<Real>& x) {
    Matrix<Real> hess = ddH(x) - x.dot(dH(x)) * Matrix<Real>::Identity(N(), N());
    return hess - (hess * x) * x.transpose() / x.squaredNorm();
  }
};

using Rng = randutils::random_generator<pcg32>;
using Real = double;

int main(int argc, char* argv[]) {
  unsigned N = 10;
  unsigned M = 10;
  Real σ = 1;

  int opt;

  while ((opt = getopt(argc, argv, "N:M:s:")) != -1) {
    switch (opt) {
    case 'N':
      N = (unsigned)atof(optarg);
      break;
    case 'M':
      M = (unsigned)atof(optarg);
      break;
    case 's':
      σ = atof(optarg);
      break;
    default:
      exit(1);
    }
  }

  Rng r;

  Model<Real> leastSquares(σ, N, M, r.engine());

  Vector<Real> x = Vector<Real>::Zero(N);
  x(0) = N;

  std::cout << leastSquares.H(x) << std::endl;
  std::cout << leastSquares.∇H(x) << std::endl;
  std::cout << leastSquares.HessH(x) << std::endl;

  return 0;
}
