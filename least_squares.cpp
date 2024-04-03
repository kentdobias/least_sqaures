#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <getopt.h>

#include "pcg-cpp/include/pcg_random.hpp"
#include "randutils/randutils.hpp"

using Rng = randutils::random_generator<pcg32>;

using Real = float;
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

  Tensor operator+(const Eigen::Tensor<Real, 3>& J) const {
    return Eigen::Tensor<Real, 3>::operator+(J);
  }
};

Matrix operator*(const Eigen::Matrix<Real, 1, Eigen::Dynamic>& x, const Tensor& J) {
  const std::array<Eigen::IndexPair<int>, 1> ip00 = {Eigen::IndexPair<int>(0, 0)};
  const Eigen::Tensor<Real, 1> xT = Eigen::TensorMap<const Eigen::Tensor<Real, 1>>(x.data(), x.size());
  const Eigen::Tensor<Real, 2> JxT = J.contract(xT, ip00);
  return Eigen::Map<const Matrix>(JxT.data(), J.dimension(1), J.dimension(2));
}

class Tensor4 : public Eigen::Tensor<Real, 4> {
  using Eigen::Tensor<Real, 4>::Tensor;

public:
  Eigen::Tensor<Real, 3> operator*(const Vector& x) const {
    const std::array<Eigen::IndexPair<int>, 1> ip30 = {Eigen::IndexPair<int>(3, 0)};
    const Eigen::Tensor<Real, 1> xT = Eigen::TensorMap<const Eigen::Tensor<Real, 1>>(x.data(), x.size());
    return contract(xT, ip30);
  }
};

Vector normalize(const Vector& x) {
  return x * sqrt((Real)x.size() / x.squaredNorm());
}

class Model {
public:
  unsigned N;
  unsigned M;

  Model(unsigned N, unsigned M) : N(N), M(M) {}

  virtual std::tuple<Real, Vector, Matrix> HdHddH(const Vector& x) const {
    return {0, Vector::Zero(N), Matrix::Zero(N, N)};
  }

  std::tuple<Real, Vector, Matrix> hamGradHess(const Vector& x) const {
    auto [H, dH, ddH] = HdHddH(x);

    Vector gradH = dH - dH.dot(x) * x / (Real)N;
    Matrix hessH = ddH - (dH * x.transpose() + x.dot(dH) * Matrix::Identity(N, N) + (ddH * x) * x.transpose()) / (Real)N  + 2.0 * x * x.transpose();

    return {H, gradH, hessH};
  }

  Real getHamiltonian(const Vector& x) const {
    Real H;
    std::tie(H, std::ignore, std::ignore) = HdHddH(x);
    return H;
  }

  Vector spectrum(const Vector& x) const {
    Matrix hessH;
    std::tie(std::ignore, std::ignore, hessH) = hamGradHess(x);
    Eigen::EigenSolver<Matrix> eigenS(hessH);
    return eigenS.eigenvalues().real();
  }
};

class QuadraticModel : public Model {
private:
  Tensor J;
  Matrix A;
  Vector b;

public:
  QuadraticModel(unsigned N, unsigned M, Rng& r, double μ1, double μ2, double μ3) : Model(N, M), J(M, N, N), A(M, N), b(M) {
    for (unsigned k = 0; k < N; k++) {
      for (unsigned j = 0; j < N; j++) {
        for (unsigned i = 0; i < M; i++) {
          J(i, j, k) = r.variate<Real, std::normal_distribution>(0, 2 * μ3 / N);
        }
      }
    }

    for (Real& Aij : A.reshaped()) {
      Aij = r.variate<Real, std::normal_distribution>(0, μ2 / sqrt(N));
    }

    for (Real& bi : b) {
      bi = r.variate<Real, std::normal_distribution>(0, μ1);
    }
  }

  std::tuple<Vector, Matrix, const Tensor&> VdVddV(const Vector& x) const {
    Matrix Jx = J * x;
    Vector V = b + (A + 0.5 * Jx) * x;
    Matrix dV = A + Jx;

    return {V, dV, J};
  }

  std::tuple<Real, Vector, Matrix> HdHddH(const Vector& x) const override {
    auto [V, dV, ddV] = VdVddV(x);

    Real H = 0.5 * V.squaredNorm();
    Vector dH = V.transpose() * dV;
    Matrix ddH = V.transpose() * ddV + dV.transpose() * dV;

    return {H, dH, ddH};
  }
};

class CubicModel : public Model {
private:
  Tensor4 J3;
  Tensor J2;
  Matrix A;
  Vector b;

public:
  CubicModel(unsigned N, unsigned M, Rng& r, double μ1, double μ2, double μ3, double μ4) : Model(N, M), J3(M, N, N, N), J2(M, N, N), A(M, N), b(M) {
    for (unsigned i = 0; i < M; i++) {
      for (unsigned j = 0; j < N; j++) {
        for (unsigned k = 0; k < N; k++) {
          for (unsigned l = 0; l < N; l++) {
            J3(i, j, k, l) = (6 * μ4 / pow(N, 1.5)) * r.variate<Real, std::normal_distribution>();
          }
        }
      }
    }

    for (unsigned k = 0; k < N; k++) {
      for (unsigned j = 0; j < N; j++) {
        for (unsigned i = 0; i < M; i++) {
          J2(i, j, k) = r.variate<Real, std::normal_distribution>(0, 2 * μ3 / N);
        }
      }
    }

    for (Real& Aij : A.reshaped()) {
      Aij = r.variate<Real, std::normal_distribution>(0, μ2 / sqrt(N));
    }

    for (Real& bi : b) {
      bi = r.variate<Real, std::normal_distribution>(0, μ1);
    }
  }


  std::tuple<Vector, Matrix, Tensor> VdVddV(const Vector& x) const {
    Tensor J3x = J3 * x;
    Matrix J3xx = J3x * x;
    Matrix J2x = J2 * x;
    Vector V1 = (A + 0.5 * J2x + J3xx / 6.0) * x;
    Matrix dV = A + J2x + 0.5 * J3xx;

    return {b + V1, dV, J2 + J3x};
  }

  std::tuple<Real, Vector, Matrix> HdHddH(const Vector& x) const override {
    auto [V, dV, ddV] = VdVddV(x);

    Real H = 0.5 * V.squaredNorm();
    Vector dH = V.transpose() * dV;
    Matrix ddH = V.transpose() * ddV + dV.transpose() * dV;

    return {H, dH, ddH};
  }
};

Vector findMinimum(const Model& M, const Vector& x0, Real ε = 1e-12) {
  Vector x = x0;
  Real λ = 100;

  auto [H, g, m] = M.hamGradHess(x0);

  while (g.norm() / M.N > ε && λ < 1e8) {
    Vector dx = (m + λ * (Matrix)m.diagonal().cwiseAbs().asDiagonal()).partialPivLu().solve(g);
    dx -= x.dot(dx) * x / M.N;
    Vector xNew = normalize(x - dx);

    auto [HNew, gNew, mNew] = M.hamGradHess(xNew);

    if (HNew < H) {
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

Vector metropolisStep(const Model& M, const Vector& x0, Real β, Rng& r, Real ε = 1) {
  Vector Δx(M.N);

  for (Real& Δxᵢ : Δx) {
    Δxᵢ = ε * r.variate<Real, std::normal_distribution>();
  }

  Vector xNew = normalize(x0 + Δx);

  Real Hold = M.getHamiltonian(x0);
  Real Hnew = M.getHamiltonian(xNew);

  if (exp(-β * (Hnew - Hold)) > r.uniform<Real>(0.0, 0.1)) {
    return xNew;
  } else {
    return x0;
  }
}

Vector metropolisSweep(const Model& M, const Vector& x0, Real β, Rng& r, Real ε = 1) {
  Vector x = x0;

  for (unsigned i = 0; i < M.N; i++) {
    x = metropolisStep(M, x, β, r, ε);
  }

  return x;
}

int main(int argc, char* argv[]) {
  unsigned N = 10;
  Real α = 1;
  Real σ = 1;
  Real A = 1;
  Real J = 1;
  Real β = 1;
  unsigned sweeps = 10;
  unsigned samples = 10;

  int opt;

  while ((opt = getopt(argc, argv, "N:a:s:A:J:b:S:n:")) != -1) {
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
    case 'b':
      β = atof(optarg);
      break;
    case 'S':
      sweeps = atoi(optarg);
      break;
    case 'n':
      samples = atoi(optarg);
      break;
    default:
      exit(1);
    }
  }

  unsigned M = (unsigned)(α * N);

  Rng r;

  Vector x = Vector::Zero(N);
  x(0) = sqrt(N);

  std::cout << N << " " << α << " " << β;

  for (unsigned sample = 0; sample < samples; sample++) {
    QuadraticModel leastSquares(N, M, r, σ, A, J);
    for (unsigned i = 0; i < sweeps; i++) {
      x = metropolisSweep(leastSquares, x, β, r);
    }
    x = findMinimum(leastSquares, x);
    std::cout << " " << leastSquares.getHamiltonian(x) / N;
  }

  std::cout << std::endl;

  return 0;
}
