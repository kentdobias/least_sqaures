#include <getopt.h>
#include <iomanip>

#include "pcg-cpp/include/pcg_random.hpp"
#include "randutils/randutils.hpp"

#include "eigen/Eigen/Dense"
#include "eigen/unsupported/Eigen/CXX11/Tensor"
#include "eigen/unsupported/Eigen/CXX11/TensorSymmetry"

using Rng = randutils::random_generator<pcg32>;

using Real = double;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

class Tensor : public Eigen::Tensor<Real, 3> {
  using Eigen::Tensor<Real, 3>::Tensor;

public:
  Matrix operator*(const Vector& x) const {
    std::array<Eigen::IndexPair<int>, 1> ip20 = {Eigen::IndexPair<int>(2, 0)};
    const Eigen::Tensor<Real, 1> xT = Eigen::TensorMap<const Eigen::Tensor<Real, 1>>(x.data(), x.size());
    const Eigen::Tensor<Real, 2> JxT = contract(xT, ip20);
    return Eigen::Map<const Matrix>(JxT.data(), dimension(0), dimension(1));
  }
};

Matrix operator*(const Eigen::Matrix<Real, 1, Eigen::Dynamic>& x, const Tensor& J) {
  std::array<Eigen::IndexPair<int>, 1> ip00 = {Eigen::IndexPair<int>(0, 0)};
  const Eigen::Tensor<Real, 1> xT = Eigen::TensorMap<const Eigen::Tensor<Real, 1>>(x.data(), x.size());
  const Eigen::Tensor<Real, 2> JxT = J.contract(xT, ip00);
  return Eigen::Map<const Matrix>(JxT.data(), J.dimension(1), J.dimension(2));
}

Vector normalize(const Vector& x) {
  return x * sqrt(x.size() / x.squaredNorm());
}

Vector ∂HFromV∂V(const Vector& V, const Matrix& ∂V) {
  return V.transpose() * ∂V;
}

Vector VFromABJx(const Vector& b, const Matrix& A, const Matrix& Jx, const Vector& x) {
  return b + (A + 0.5 * Jx) * x;
}

class QuadraticModel {
private:
  Tensor J;
  Matrix A;
  Vector b;

  std::tuple<Vector, Matrix, const Tensor&> V_∂V_∂∂V(const Vector& x) const {
    Matrix Jx = J * x;
    Vector V = VFromABJx(b, A, Jx, x);
    Matrix ∂V = A + Jx;
    return {V, ∂V, J};
  }

  std::tuple<Vector, Matrix> ∂H_∂∂H(const Vector& x) const {
    auto [V, ∂V, ∂∂V] = V_∂V_∂∂V(x);
    Vector ∂H = ∂HFromV∂V(V, ∂V);
    Matrix ∂∂H = V.transpose() * ∂∂V + ∂V.transpose() * ∂V;
    return {∂H, ∂∂H};
  }

public:
  unsigned N;
  unsigned M;

  QuadraticModel(unsigned N, unsigned M, Real σ², Real μA, Real μJ, Rng& r) : J(M, N, N), A(M, N), b(M), N(N), M(M) {
    Eigen::StaticSGroup<Eigen::Symmetry<1,2>> sym23;

    for (unsigned k = 0; k < N; k++) {
      for (unsigned j = k; j < N; j++) {
        for (unsigned i = 0; i < M; i++) {
          sym23(J, i, j, k) = r.variate<Real, std::normal_distribution>(0, sqrt(2 * μJ) / N);
        }
      }
    }

    for (Real& Aij : A.reshaped()) {
      Aij = r.variate<Real, std::normal_distribution>(0, sqrt(μA / N));
    }

    for (Real& bi : b) {
      bi = r.variate<Real, std::normal_distribution>(0, sqrt(σ²));
    }
  }

  Real H(const Vector& x) const {
    Vector V = VFromABJx(b, A, J * x, x);
    return 0.5 * V.squaredNorm();
  }

  Vector ∇H(const Vector& x) const {
    auto [V, ∂V, ∂∂V] = V_∂V_∂∂V(x);
    Vector ∂H = ∂HFromV∂V(V, ∂V);
    return ∂H - (∂H.dot(x) / x.squaredNorm()) * x;
  }

  Matrix HessH(const Vector& x) const {
    auto [∂H, ∂∂H] = ∂H_∂∂H(x);
    Matrix P = Matrix::Identity(N, N) - x * x.transpose() / x.squaredNorm();
    return P * ∂∂H * P.transpose() - (x.dot(∂H) / N) * Matrix::Identity(N, N);
  }

  Vector spectrum(const Vector& x) const {
    Matrix hessH  = HessH(x);
    Eigen::EigenSolver<Matrix> eigenS(hessH);
    return eigenS.eigenvalues().real();
  }

  Real λmax(const Vector& x) const {
    return spectrum(x).maxCoeff();
  }
};

Vector gradientAscent(const QuadraticModel& M, const Vector& x₀, Real ε = 1e-13) {
  Vector xₜ = x₀;
  Real Hₜ = M.H(x₀);
  Real α = 1;
  Real m;
  Vector ∇H;

  while (
    ∇H = M.∇H(xₜ), m = ∇H.squaredNorm(),
    m / M.N > ε
  ) {
    Vector xₜ₊₁;
    Real Hₜ₊₁;

    while(
      xₜ₊₁ = normalize(xₜ + α * ∇H), Hₜ₊₁ = M.H(xₜ₊₁),
      Hₜ₊₁ < Hₜ + 0.5 * α * m
    ) {
      α /= 2;
    }

    xₜ = xₜ₊₁;
    Hₜ = Hₜ₊₁;
    α *= 1.25;
  }

  return xₜ;
}

Vector messagePassing(const QuadraticModel& M, const Vector& x₀) {
  Vector σ = x₀ / x₀.norm();

  for (unsigned i = 0; i < M.N; i++) {
    Vector ∇H = M.∇H(σ);
    Vector v = ∇H / ∇H.norm();

    σ += v;
  }

  return normalize(σ);
}

int main(int argc, char* argv[]) {
  unsigned N = 10;
  Real α = 1;
  Real σ² = 1;
  Real μA = 1;
  Real μJ = 1;
  unsigned samples = 10;

  int opt;

  while ((opt = getopt(argc, argv, "N:a:s:A:J:n:")) != -1) {
    switch (opt) {
    case 'N':
      N = (unsigned)atof(optarg);
      break;
    case 'a':
      α = atof(optarg);
      break;
    case 's':
      σ² = atof(optarg);
      break;
    case 'A':
      μA = atof(optarg);
      break;
    case 'J':
      μJ = atof(optarg);
      break;
    case 'n':
      samples = atoi(optarg);
      break;
    default:
      exit(1);
    }
  }

  unsigned M = std::round(α * N);

  Rng r;

  Vector x = Vector::Zero(N);
  x(0) = sqrt(N);

  std::cout << std::setprecision(15);

  for (unsigned sample = 0; sample < samples; sample++) {
    QuadraticModel* ls = new QuadraticModel(N, M, σ², μA, μJ, r);
    Vector xGD = gradientAscent(*ls, x);
    std::cout << ls->H(xGD) / N << " " << ls->λmax(xGD) << " ";
    delete ls;

    ls = new QuadraticModel(N, M, σ², μA, μJ, r);
    Vector xMP = messagePassing(*ls, x);
    xMP = gradientAscent(*ls, xMP);
    std::cout << ls->H(xMP) / N << " " << ls->λmax(xMP) << std::endl;
    delete ls;
  }

  return 0;
}
