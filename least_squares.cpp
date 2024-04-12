#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/CXX11/TensorSymmetry>
#include <getopt.h>

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

Vector makeTangent(const Vector& v, const Vector& x) {
  return v - (v.dot(x) / x.size()) * x;
}

Real HFromV(const Vector& V) {
  return 0.5 * V.squaredNorm();
}

Vector dHFromVdV(const Vector& V, const Matrix& dV) {
  return V.transpose() * dV;
}

Vector VFromABJ(const Vector& b, const Matrix& A, const Matrix& Jx, const Vector& x) {
  return b + (A + 0.5 * Jx) * x;
}

class QuadraticModel {
private:
  Tensor J;
  Matrix A;
  Vector b;

public:
  unsigned N;
  unsigned M;
  QuadraticModel(unsigned N, unsigned M, Rng& r, double μ1, double μ2, double μ3) : N(N), M(M), J(M, N, N), A(M, N), b(M) {
    Eigen::StaticSGroup<Eigen::Symmetry<1,2>> ssym1;
    for (unsigned k = 0; k < N; k++) {
      for (unsigned j = k; j < N; j++) {
        for (unsigned i = 0; i < M; i++) {
          ssym1(J, i, j, k) = r.variate<Real, std::normal_distribution>(0, sqrt(2) * μ3 / N);
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
    Vector V = VFromABJ(b, A, Jx, x);
    Matrix dV = A + Jx;

    return {V, dV, J};
  }

  std::tuple<Real, Vector, Matrix> HdHddH(const Vector& x) const {
    auto [V, dV, ddV] = VdVddV(x);

    Real H = HFromV(V);
    Vector dH = dHFromVdV(V, dV);
    Matrix ddH = V.transpose() * ddV + dV.transpose() * dV;

    return {H, dH, ddH};
  }

  /* Unfortunately benchmarking indicates that ignorning entries of a returned tuple doesn't result
   * in those execution paths getting optimized out. It is much more efficient to compute the
   * energy alone when only the energy is needed.
   */
  Real getHamiltonian(const Vector& x) const {
    return HFromV(VFromABJ(b, A, J * x, x));
  }

  std::tuple<Real, Vector> getHamGrad(const Vector& x) const {
    Vector V;
    Matrix dV;
    std::tie(V, dV, std::ignore) = VdVddV(x);

    Real H = HFromV(V);
    Vector dH = makeTangent(dHFromVdV(V, dV), x);

    return {H, dH};
  }

  std::tuple<Real, Vector, Matrix> hamGradHess(const Vector& x) const {
    auto [H, dH, ddH] = HdHddH(x);

    Real dHx = dH.dot(x) / N;

    Vector gradH = dH - dHx * x;
    Matrix hessH = ddH - dHx * Matrix::Identity(N, N) - ((dH + ddH * x) / N - 2 * x) * x.transpose();

    return {H, gradH, hessH};
  }

  Vector spectrum(const Vector& x) const {
    Matrix hessH;
    std::tie(std::ignore, std::ignore, hessH) = hamGradHess(x);
    Eigen::EigenSolver<Matrix> eigenS(hessH);
    return eigenS.eigenvalues().real();
  }
};

Vector gradientDescent(const QuadraticModel& M, const Vector& x0, Real ε = 1e-7) {
  Vector x = x0;
  Real λ = 10;

  auto [H, g] = M.getHamGrad(x);

  while (g.norm() / M.N > ε && λ > ε) {
    Real HNew;
    Vector xNew, gNew;

    while(
      xNew = normalize(x + λ * g),
      std::tie(HNew, gNew) = M.getHamGrad(xNew),
      HNew < H && λ > ε
    ) {
      λ /= 1.5;
    }

    x = xNew;
    H = HNew;
    g = gNew;

    λ *= 2;
  }

  return x;
}

Vector findMinimum(const QuadraticModel& M, const Vector& x0, Real ε = 1e-5) {
  Vector x = x0;
  Real λ = 100;

  auto [H, g, m] = M.hamGradHess(x0);

  while (λ * ε < 1) {
    Vector dx = (m - λ * (Matrix)m.diagonal().cwiseAbs().asDiagonal()).partialPivLu().solve(g);
    Vector xNew = normalize(x - makeTangent(dx, x));
    Real HNew = M.getHamiltonian(xNew);

    if (HNew > H) {
      x = xNew;
      std::tie(H, g, m) = M.hamGradHess(xNew);

      λ /= 2;
    } else {
      λ *= 1.5;
    }
  }

  return x;
}

Vector findSaddle(const QuadraticModel& M, const Vector& x0, Real ε = 1e-12) {
  Vector x = x0;
  Vector g;
  Matrix m;

  while (std::tie(std::ignore, g, m) = M.hamGradHess(x), g.norm() / M.N > ε) {
    Vector dx = m.partialPivLu().solve(g);
    dx -= (x.dot(dx) / M.N) * x;
    x = normalize(x - dx);
  }

  return x;
}

Vector metropolisSweep(const QuadraticModel& M, const Vector& x0, Real β, Rng& r, unsigned sweeps = 1, Real ε = 1) {
  Vector x = x0;
  Real H = M.getHamiltonian(x);

  for (unsigned j = 0; j < sweeps; j++) {
    Real rate = 0;

    for (unsigned i = 0; i < M.N; i++) {
      Vector xNew = x;

      for (Real& xNewᵢ : xNew) {
        xNewᵢ += ε * r.variate<Real, std::normal_distribution>();
      }

      xNew = normalize(xNew);

      Real Hnew = M.getHamiltonian(xNew);

      if (exp(-β * (Hnew - H)) > r.uniform<Real>(0.0, 0.1)) {
        x = xNew;
        H = Hnew;
        rate++;
      }
    }

    if (rate / M.N < 0.5) {
      ε /= 1.5;
    } else {
      ε *= 1.5;
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
    if (β != 0) {
      x = metropolisSweep(leastSquares, x, β, r, sweeps);
    }
    x = findMinimum(leastSquares, x);
    std::cout << " " << leastSquares.getHamiltonian(x) / N << std::flush;
  }

  std::cout << std::endl;

  return 0;
}
