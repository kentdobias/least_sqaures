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

Vector dHFromVdV(const Vector& V, const Matrix& ∂V) {
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

  std::tuple<Real, Vector, Matrix> H_∂H_∂∂H(const Vector& x) const {
    auto [V, ∂V, ∂∂V] = V_∂V_∂∂V(x);

    Real H = HFromV(V);
    Vector ∂H = dHFromVdV(V, ∂V);
    Matrix ∂∂H = V.transpose() * ∂∂V + ∂V.transpose() * ∂V;

    return {H, ∂H, ∂∂H};
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

  std::tuple<Real, Vector> getHamGrad(const Vector& x) const {
    auto [V, dV, ddV] = V_∂V_∂∂V(x);

    Real H = HFromV(V);
    Vector ∂H = dHFromVdV(V, dV);
    Vector ∇H = makeTangent(∂H, x);

    return {H, ∇H};
  }

  std::tuple<Real, Vector, Matrix> getHamGradHess(const Vector& x) const {
    auto [H, ∂H, ∂∂H] = H_∂H_∂∂H(x);

    Vector ∇H = makeTangent(∂H, x);
    Matrix HessH = ∂∂H + (2 * x - (∂H + ∂∂H * x) / N) * x.transpose()
      - (x.dot(∂H) / N) * Matrix::Identity(N, N);

    return {H, ∇H, HessH};
  }

  Vector spectrum(const Vector& x) const {
    Matrix hessH;
    std::tie(std::ignore, std::ignore, hessH) = getHamGradHess(x);
    Eigen::EigenSolver<Matrix> eigenS(hessH);
    return eigenS.eigenvalues().real();
  }
};

Vector gradientDescent(const QuadraticModel& M, const Vector& x0, Real ε = 1e-7) {
  Vector x = x0;
  Real λ = 10;

  auto [H, ∇H] = M.getHamGrad(x);

  while (∇H.norm() / M.N > ε && λ > ε) {
    Real HNew;
    Vector xNew, gNew;

    while(
      xNew = normalize(x + λ * ∇H),
      HNew = M.getHamiltonian(xNew),
      HNew < H && λ > ε
    ) {
      λ /= 1.5;
    }

    x = xNew;
    std::tie(H, ∇H) = M.getHamGrad(xNew);

    λ *= 2;
  }

  return x;
}

Vector levenbergMarquardt(const QuadraticModel& M, const Vector& x0, Real ε = 1e-5) {
  Vector x = x0;
  Real λ = 100;

  auto [H, ∇H, HessH] = M.getHamGradHess(x0);

  while (λ * ε < 1) {
    Vector Δx = (HessH - λ * (Matrix)HessH.diagonal().cwiseAbs().asDiagonal()).partialPivLu().solve(∇H);
    Vector xNew = normalize(x - makeTangent(Δx, x));
    Real HNew = M.getHamiltonian(xNew);

    if (HNew > H) {
      x = xNew;
      std::tie(H, ∇H, HessH) = M.getHamGradHess(xNew);

      λ /= 1.5;
    } else {
      λ *= 1.25;
    }
  }

  return x;
}

Vector subagAlgorithm(const QuadraticModel& M, Rng& r, unsigned k) {
  Vector σ = Vector::Zero(M.N);
  unsigned axis = r.variate<unsigned, std::uniform_int_distribution>(0, M.N - 1);
  σ(axis) = sqrt(M.N / k);

  for (unsigned i = 0; i < k; i++) {
    auto [H, ∇H] = M.getHamGrad(σ);
    Vector v = ∇H / ∇H.norm();

    σ += sqrt(M.N/k) * v;
  }

  return normalize(σ);
}

int main(int argc, char* argv[]) {
  unsigned N = 10;
  Real α = 1;
  Real σ = 1;
  Real A = 1;
  Real J = 1;
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
      σ = atof(optarg);
      break;
    case 'A':
      A = atof(optarg);
      break;
    case 'J':
      J = atof(optarg);
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

  for (unsigned sample = 0; sample < samples; sample++) {
    QuadraticModel leastSquares(N, M, r, σ, A, J);
    x = gradientDescent(leastSquares, x);
    std::cout << leastSquares.getHamiltonian(x) / N;

    leastSquares = QuadraticModel(N, M, r, σ, A, J);
    x = subagAlgorithm(leastSquares, r, N);
    x = gradientDescent(leastSquares, x);
    std::cout << " " << leastSquares.getHamiltonian(x) / N << std::endl;
  }

  return 0;
}
