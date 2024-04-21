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
  return v - (v.dot(x) / x.squaredNorm()) * x;
}

Real HFromV(const Vector& V) {
  return 0.5 * V.squaredNorm();
}

Vector ∂HFromV∂V(const Vector& V, const Matrix& ∂V) {
  return V.transpose() * ∂V;
}

Vector ∇HFromV∂Vx(const Vector& V, const Matrix& ∂V, const Vector& x) {
  return makeTangent(∂HFromV∂V(V, ∂V), x);
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

  QuadraticModel(unsigned N, unsigned M, Rng& r, Real μ1, Real μ2, Real μ3) : J(M, N, N), A(M, N), b(M), N(N), M(M) {
    Eigen::StaticSGroup<Eigen::Symmetry<1,2>> sym23;

    for (unsigned k = 0; k < N; k++) {
      for (unsigned j = k; j < N; j++) {
        for (unsigned i = 0; i < M; i++) {
          sym23(J, i, j, k) = r.variate<Real, std::normal_distribution>(0, sqrt(2 * μ3) / N);
        }
      }
    }

    for (Real& Aij : A.reshaped()) {
      Aij = r.variate<Real, std::normal_distribution>(0, sqrt(μ2 / N));
    }

    for (Real& bi : b) {
      bi = r.variate<Real, std::normal_distribution>(0, sqrt(μ1));
    }
  }

  Real getHamiltonian(const Vector& x) const {
    return HFromV(VFromABJx(b, A, J * x, x));
  }

  Vector getGradient(const Vector& x) const {
    auto [V, ∂V, ∂∂V] = V_∂V_∂∂V(x);

    return ∇HFromV∂Vx(V, ∂V, x);
  }

  std::tuple<Real, Vector> getHamGrad(const Vector& x) const {
    auto [V, ∂V, ∂∂V] = V_∂V_∂∂V(x);

    Real H = HFromV(V);
    Vector ∇H = ∇HFromV∂Vx(V, ∂V, x);

    return {H, ∇H};
  }

  Matrix getHessian(const Vector& x) const {
    auto [∂H, ∂∂H] = ∂H_∂∂H(x);

    Matrix P = Matrix::Identity(N, N) - x * x.transpose() / x.squaredNorm();
    Matrix HessH = P * ∂∂H * P.transpose() - (x.dot(∂H) / N) * Matrix::Identity(N, N);

    return HessH;
  }

  Vector spectrum(const Vector& x) const {
    Matrix HessH  = getHessian(x);
    Eigen::EigenSolver<Matrix> eigenS(HessH);
    return eigenS.eigenvalues().real();
  }

  Real maxEigenvalue(const Vector& x) const {
    return spectrum(x).maxCoeff();
  }
};

Vector gradientAscent(const QuadraticModel& M, const Vector& x0, Real ε = 1e-13) {
  Vector x = x0;
  Real α = 1;
  Real H = M.getHamiltonian(x0);
  Real m;
  Vector ∇H;

  while (
    ∇H = M.getGradient(x),
    m = ∇H.squaredNorm(),
    m / M.N > ε
  ) {
    Real HNew;
    Vector xNew;

    while(
      xNew = normalize(x + α * ∇H),
      HNew = M.getHamiltonian(xNew),
      HNew < H + 0.5 * α * m
    ) {
      α /= 2;
    }

    x = xNew;
    H = HNew;
    α *= 1.25;
  }

  return x;
}

Vector subagAlgorithm(const QuadraticModel& M, Rng& r, unsigned k) {
  Vector σ = Vector::Zero(M.N);
  unsigned axis = r.variate<unsigned, std::uniform_int_distribution>(0, M.N - 1);
  σ(axis) = sqrt(M.N / k);

  for (unsigned i = 0; i < k; i++) {
    Vector ∇H = M.getGradient(σ);
    Vector v = ∇H / ∇H.norm();

    σ += sqrt(M.N/k) * v;
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

  for (unsigned sample = 0; sample < samples; sample++) {
    QuadraticModel ls(N, M, r, σ², μA, μJ);
    Vector xGD = gradientAscent(ls, x);
    std::cout << ls.getHamiltonian(xGD) / N << " " << ls.maxEigenvalue(xGD) << " ";

    ls = QuadraticModel(N, M, r, σ², μA, μJ);
    Vector xMP = subagAlgorithm(ls, r, N);
    xMP = gradientAscent(ls, xMP);
    std::cout << ls.getHamiltonian(xMP) / N << " " << ls.maxEigenvalue(xMP) << std::endl;
  }

  return 0;
}
