using Plots            # To use plot
using GaussQuadrature  # To use legendre
using SparseArrays     # To use spzeros


# Approximates the Integral of a given function in the interval [-1:1]
function gaussian_quadrature(f, ngp)

  # Initializing P and W according to the number of Gauss points
  P, W = legendre(ngp)
  sum = 0

  for j in 1:ngp
    sum = sum + W[j] * f(P[j])
  end

  return sum
end

# Initializes the LG matrix
function init_LG_matrix(ne)
  LG = zeros(Int, 2,ne)

    for j in 1:ne
      LG[1,j] = j
      LG[2,j] = j + 1
    end

  return LG
end

# Initializes the EQ vector and the m variable
function init_EQ_vector_and_m(ne)
  # Initializing m and EQ
  m = ne - 1
  EQ = zeros(Int, ne+1)

  # Computing the first element of EQ
  EQ[1] = m + 1

  # Computing the mid elements of EQ
  for i in 1:m+1
    EQ[i+1] = i
  end

  # Computing the last element of EQ
  EQ[ne+1] = m + 1

  return EQ, m
end

# Initializes the Ke matrix
function init_Ke_matrix(ne, alpha, beta, gamma)
  h = 1 / ne
  Ke = zeros(2,2)

  for a in 1:2
    for b in 1:2
      Ke[a,b] = (alpha * 2 / h) * gaussian_quadrature((qsi) -> d_phi(a, qsi) * d_phi(b, qsi), 2) + (beta * h / 2) * gaussian_quadrature((qsi) -> phi(a, qsi) * phi(b, qsi), 2) + gamma * gaussian_quadrature((qsi) -> d_phi(b, qsi) * phi(a, qsi), 2)
    end
  end

  return Ke
end

# Initializes the K matrix
function init_K_matrix(ne, EQ, LG, alpha, beta, gamma, m)

  # Initializing K and Ke matrices
  K = spzeros(m+1,m+1)
  Ke = init_Ke_matrix(ne, alpha, beta, gamma)

  for e in 1:ne
    for a in 1:2
      i = Int(EQ[LG[a, e]])
      for b in 1:2
        j = Int(EQ[LG[b, e]])
        K[i,j] += Ke[a,b]
      end
    end
  end

  # removes the last line and column
  return K[1:m, 1:m]
end

# Initializes the F vector
function init_F_vector(f, ne, EQ, LG, m, alpha, beta, gamma)

  # Initializes the Fe vector
  function init_Fe_vector(f, ne, e)
    Fe = zeros(2)
    h = 1 / ne

    for a in 1:2
      Fe[a] = (h / 2) * gaussian_quadrature((qsi) -> f(qsi_to_x(qsi, e, h)) *  phi(a, qsi), 5)
    end

    return Fe
  end

  # Initializing the F vector and the variable h
  h = 1 / ne
  F = zeros(m+1)

  for e in 1:ne
    Fe = init_Fe_vector(f, ne, e)
    for a in 1:2
      i = EQ[LG[a,e]]
      F[i] += Fe[a]
    end
  end

  Ke = init_Ke_matrix(ne, alpha, beta, gamma)

  F[1] -= p * Ke[1,2]
  F[m] -= q * Ke[1,2]

  # removes the last line
  return F[1:m]
end

# Generalized phi function
function phi(number, qsi)
  return [((1 - qsi) / 2), ((1 + qsi) / 2)][number]
end

# Generalized derivative of the phi function
function d_phi(number, qsi)
  return [(-1 / 2), (1 / 2)][number]
end

# Converts the interval from [x_i-1 , xi+1] to [-1, 1]
function qsi_to_x(qsi, i, h)
  return (h / 2) * (qsi + 1) + 0 + (i - 1)*h
end

function solve_system(ne, alpha, beta, gamma, f, u, p, q)
  # Initializing matrices, vectors and variables
  EQ, m = init_EQ_vector_and_m(ne)
  LG    = init_LG_matrix(ne)
  K     = init_K_matrix(ne, EQ, LG, alpha, beta, gamma, m)
  F     = init_F_vector(f, ne, EQ, LG, m, alpha, beta, gamma)
  return K \ F
end

# Plots the exact and inexact graphs, as well as the absolute and relative errors
function plot_comparison(ne, alpha, beta, gamma, f, u, p, q)
  # Initializing variables
  h = 1 / ne
  xs = [h * i for i in 1:ne-1]
  Cs = solve_system(ne, alpha, beta, gamma, f, u, p, q)

  # Including the boundary conditions in both xs and Cs
  ext_xs = [0; xs; 1]
  ext_Cs = [p; Cs; q]

  # Plotting the exact function and our approximation
  plt = plot(u, 0, 1, label = "u(x)", size=(800, 800))
  plot!(plt, ext_xs, ext_Cs, seriestype = :scatter, label = "Approximation", xlabel = "x", ylabel = "Approximation for u(x)", size=(800, 800))

  # Saving the graph
  savefig("approximation-graph.png")
end

# Plots the graph of errors according to the varying of n
function error_analysis(lb, ub)

  # Computes the error according to ne
  function gauss_error(u, cs, ne, EQ, LG)
    sum = 0
    h = 1 / ne

    # Including 0 so that the EQ-LG will not consider the first and the last phi function
    extended_cs = [cs; 0]

    # Computing the error
    for e in 1:ne
      sum = sum + gaussian_quadrature((qsi) -> (u(qsi_to_x(qsi, e, h)) - (extended_cs[EQ[LG[1,e]]] * phi(1, qsi)) - (extended_cs[EQ[LG[2,e]]] * phi(2, qsi)))^2, 5)
    end

    return sqrt(sum * (h / 2))
  end

  # Initializing the vectors
  errors = zeros(ub - lb + 1)
  nes = [(1 << i) - 1 for i in lb:ub]
  hs = [1 / nes[i - lb + 1] for i in lb:ub]

  # Computing the errors varying according to the variation of h
  for i in lb:ub
    ne = nes[i-lb+1]
    EQ, m = init_EQ_vector_and_m(ne)
    LG = init_LG_matrix(ne)
    Cs = solve_system(ne, alpha, beta, gamma, f, u, p, q)
    e = gauss_error(u, Cs, ne, EQ, LG)
    errors[i-lb+1] = e
  end

  # Plotting the errors in the graphic in a log scale
  plot(hs, errors, seriestype = :scatter, label = "Error convergence ",
       xlabel = "h", ylabel = "error", size=(800, 800), xscale=:log10, yscale=:log10,
       markercolor = :blue)
  plot!(hs, errors, seriestype = :line, label = "", linewidth = 2, linecolor = :blue)
  plot!(hs, hs.^2, seriestype = :line, label = "h^2", linewidth = 2, linecolor = :red)

  # Saves the graph in a png file
  savefig("errors-convergence.png")
end


# Constants
alpha = 1
beta  = 1
gamma = 0
p     = 0
q     = 1

# Functions
# f = (x) -> alpha*pi^2 * sin(pi*x) + beta*sin(pi*x) + gamma*pi*cos(pi*x)
# u = (x) -> sin(pi * x)

f = (x) -> x
# u = (x) -> x + 1 / (exp(1) - exp(-1)) * ((exp(1) + 1) * exp(-x) - (exp(-1) + 1) * exp(x))
u = (x) -> x

# Bound limits for analyzing the error convergence
lb = 2
ub = 10

# Number of elements for plotting a comparison graph and displaying the time it took to compute the approximation
ne = 8

# Testing the implementation
plot_comparison(ne, alpha, beta, gamma, f, u, p, q)
error_analysis(lb, ub)
# @time begin
#   C = solve_system(ne, alpha, beta, gamma, f, u, p, q)
# end

# TODO: fix the error_analysis function
