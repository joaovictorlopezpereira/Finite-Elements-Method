using Plots            # To use plot and gif
using GaussQuadrature  # To use legendre
using SparseArrays     # To use spzeros
using LinearAlgebra    # To use lu


# Approximates the Integral of a given function
function gaussian_quadrature(f, ngp)

  # Initializes P and W according to the number of Gauss points
  P, W = legendre(ngp)
  sum = 0

  # Calculates the sum
  for j in 1:ngp
    sum = sum + W[j] * f(P[j])
  end

  return sum
end

# Initializes the Fe vector
function init_Fe_vector(f, Nx1, Nx2, pe1, pe2)
  Fe = zeros(4)
  h1 = 1 / Nx1
  h2 = 1 / Nx2
  ngp = 5

  for a in 1:4
    Fe[a] = h1 * h2 / 4 * gaussian_quadrature((xi2) ->
                          gaussian_quadrature((xi1) -> f(x(xi1, xi2, 1, h1, h2, pe1, pe2),
                                                         x(xi1, xi2, 2, h1, h2, pe1, pe2)) *
                                          phi(a, xi1, xi2), ngp), ngp)
  end

  return Fe
end

# Initializes the Ke matrix
function init_Ke_matrix(alpha, beta, Nx1, Nx2)
  Ke = zeros(4,4)
  h1 = 1 / Nx1
  h2 = 1 / Nx2
  ngp = 2

  for a in 1:4
    for b in 1:4
      Ke[a,b] = (alpha * h2 / h1)    * gaussian_quadrature((xi2) ->
                                       gaussian_quadrature((xi1) -> d_phi(b, xi1, xi2, 1) * d_phi(a, xi1, xi2, 1), ngp), ngp) +
                (alpha * h1 / h2)    * gaussian_quadrature((xi2) ->
                                       gaussian_quadrature((xi1) -> d_phi(b, xi1, xi2, 2) * d_phi(a, xi1, xi2, 2), ngp), ngp) +
                (beta * h1 * h2 / 4) * gaussian_quadrature((xi2) ->
                                       gaussian_quadrature((xi1) ->   phi(b, xi1, xi2)    *   phi(a, xi1, xi2),    ngp), ngp)
    end
  end

  return Ke
end

# Phi function
function phi(a, xi1, xi2)
  return [(1 - xi1) * (1 - xi2) * (1 / 4);
          (1 + xi1) * (1 - xi2) * (1 / 4);
          (1 + xi1) * (1 + xi2) * (1 / 4);
          (1 - xi1) * (1 + xi2) * (1 / 4)][a]
end

# Derivative of the phi function
function d_phi(a, xi1, xi2, dxi)
  [[((-1 / 4) * (1 - xi2)),
    (( 1 / 4) * (1 - xi2)),
    (( 1 / 4) * (1 + xi2)),
    ((-1 / 4) * (1 + xi2))
  ],[
    ((-1 / 4) * (1 - xi1)),
    ((-1 / 4) * (1 + xi1)),
    (( 1 / 4) * (1 + xi1)),
    (( 1 / 4) * (1 - xi1))]][dxi][a]
end

# Converts the interval
function x(xi, eta, num, h1, h2, pe1, pe2)
  return [(h1 / 2) * (xi + 1) + pe1;
          (h2 / 2) * (eta + 1) + pe2][num]
end

# Initializes the LG matrix
function init_LG_matrix(nx1, nx2)
  ne = nx1 * nx2
  LG = fill(0, (4, ne))
  j = 0

  for i in 1:ne
    if (j % (nx1+1) == 0)
      j = j + 1
    end

    LG[1, i] = j
    LG[2, i] = j + 1
    LG[3, i] = j + nx1 + 2
    LG[4, i] = j + nx1 + 1

    j = j + 1
  end

  return LG
end

# Initializes the EQ Vector
function init_EQ_vector_and_m(nx1, nx2)
  m = (nx1 - 1) * (nx2 -1)
  EQ = fill(0, (nx2+1, nx1+1))

  # Initializes the border elements
  for i in 1:nx1+1
    EQ[1,i] = m + 1
    EQ[nx2+1, i] = m + 1
  end
  for j in 1:nx2+1
    EQ[j, 1] = m+1
    EQ[j, nx1+1] = m+1
  end

  # initializes the within elements
  k = 1
  for i in 2:nx2
    for j in 2:nx1
      EQ[i,j] = k
      k = k + 1
    end
  end

  return cat(EQ'..., dims=1), m
end

# Initializes the K matrix
function init_K_matrix(alpha, beta, Nx1, Nx2, m, EQ, LG)
  K = spzeros(m+1, m+1)
  Ke = init_Ke_matrix(alpha, beta, Nx1, Nx2)
  ne = Nx1 * Nx2

  for e in 1:ne
    for b in 1:4
      j = EQ[LG[b, e]]
      for a in 1:4
        i = EQ[LG[a, e]]
        K[i,j] += Ke[a,b]
      end
    end
  end

  return K[1:end-1, 1:end-1]
end

# Initializes the F vector
function init_F_vector(f, Nx1, Nx2, m, EQ, LG)
  F = zeros(m+1)
  h1 = 1 / Nx1
  h2 = 1 / Nx2

  for j in 1:Nx2
    pe2 = (j - 1) * h2
    for i in 1:Nx1
      pe1 = (i - 1) * h1
      Fe = init_Fe_vector(f, Nx1, Nx2, pe1, pe2)
      e = (j - 1) * Nx1 + i
      for a in 1:4
        F[EQ[LG[a,e]]] += Fe[a]
      end
    end
  end

  return F[1: end-1]
end

# Solves the system
function solve_system(alpha, beta, Nx1, Nx2, f)
  EQ, m = init_EQ_vector_and_m(Nx1, Nx2)
  LG = init_LG_matrix(Nx1, Nx2)
  K = init_K_matrix(alpha, beta, Nx1, Nx2, m, EQ, LG)
  F = init_F_vector(f, Nx1, Nx2, m, EQ, LG)
  C = K \ F
  return C
end

# Input data
alpha = 1
beta = 1
Nx1 = 4
Nx2 = 4

u     = (x1,x2) -> sin(pi * x1) * sin(pi * x2) 
ux1x1 = (x1,x2) -> -1 * pi^2 * sin(pi * x1) * sin(pi * x2)
ux2x2 = (x1,x2) -> -1 * pi^2 * sin(pi * x1) * sin(pi * x2)
f     = (x1,x2) -> (-1 * alpha * ux1x1(x1,x2)) + (-1 * alpha * ux2x2(x1,x2)) + beta * u(x1,x2)

# Testing 
EQ, m = init_EQ_vector_and_m(Nx1, Nx2)
LG = init_LG_matrix(Nx1, Nx2)
K = init_K_matrix(alpha, beta, Nx1, Nx2, m, EQ, LG)
F = init_F_vector(f, Nx1, Nx2, m, EQ, LG)

C = K \ F

display(C)