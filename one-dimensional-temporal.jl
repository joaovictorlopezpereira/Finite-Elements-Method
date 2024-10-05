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

# Initializes the LG matrix
function init_LG_matrix(ne)
  LG = zeros(Int, 2, ne)

    for j in 1:ne
      LG[1,j] = j
      LG[2,j] = j + 1
    end

  return LG
end

# Initializes the EQ vector
function init_EQ_vector_and_m(ne)
  m = ne - 1
  EQ = zeros(Int, ne+1)

  # Initializes the first element in EQ
  EQ[1] = m + 1

  # Initializes the middle elements of EQ
  for i in 1:m+1
    EQ[i+1] = i
  end

  # Initializes the last element in EQ
  EQ[ne+1] = m + 1

  return EQ, m
end

# Initializes the Ke matrix
function init_Ke_matrix(ne, alpha, beta, gamma)
  h = 1 / ne
  Ke = zeros(2,2)

  for a in 1:2
    for b in 1:2
      Ke[a,b] = (alpha * 2 / h) * gaussian_quadrature((qsi) -> d_phi(a, qsi) * d_phi(b, qsi), 2) + 
                 (beta * h / 2) * gaussian_quadrature((qsi) -> phi(a, qsi) * phi(b, qsi), 2) + 
                          gamma * gaussian_quadrature((qsi) -> d_phi(b, qsi) * phi(a, qsi), 2)
    end
  end

  return Ke
end

# Initializes the K matrix
function init_K_matrix(ne, EQ, LG, alpha, beta, gamma, m)
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

# Initializes the Fe vector
function init_Fe_vector(f, ne, e)
  Fe = zeros(2)
  h = 1 / ne

  for a in 1:2
    Fe[a] = (h / 2) * gaussian_quadrature((qsi) -> f(qsi_to_x(qsi, e, h)) *  phi(a, qsi), 5)
  end

  return Fe
end

# Initializes the F vector
function init_F_vector(f, ne, EQ, LG, m)
  h = 1 / ne
  F = zeros(m+1)
  EQoLG = EQ[LG]

  for e in 1:ne
    Fe = init_Fe_vector(f, ne, e)
    for a in 1:2
      i = EQ[LG[a,e]]
      F[i] += Fe[a]
    end
  end

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

# Computes the error according to ne
function gauss_error(u, cs, ne, EQ, LG)
  sum = 0
  h = 1 / ne

  # Includes 0 so that the EQ-LG will not compute the first and last phi function
  ext_cs = [cs ; 0]

  # Computes the error 
  for e in 1:ne
    sum = sum + gaussian_quadrature((qsi) -> (u(qsi_to_x(qsi, e, h)) - (ext_cs[EQ[LG[1,e]]] * phi(1, qsi)) - (ext_cs[EQ[LG[2,e]]] * phi(2, qsi)))^2, 5)
  end

  return sqrt(sum * (h / 2))
end

# Solves the system and returns Cns
function solve_system(ne, tau, alpha, beta, gamma, T, f, u0)
  # Initializing variables, matrices and vectors
  h = 1 / ne
  t = 0:tau:T
  N = length(t) - 1
  LG = init_LG_matrix(ne)
  EQ, m = init_EQ_vector_and_m(ne)
  K = init_K_matrix(ne, EQ, LG, alpha, beta, gamma, m)
  M = init_K_matrix(ne, EQ, LG, 0, 1, gamma, m)
  A = M + ((tau / 2) * K)
  B = M - ((tau / 2) * K)
  LU = lu(A)
  Cn = zeros(m)
  Cn_1 = zeros(m)
  Cns = Vector{Vector{Float64}}(undef, N+1)

  # Iterations for computing u by varying n (time)
  for n in 0:N
    if (n == 0)
      Cn .= u0.(h:h:1-h)
      Cn_1 = Cn
      Cns[n+1] = Cn
    else
      F = init_F_vector((x) -> f(x, (n*tau - tau/2)), ne, EQ, LG, m)
      Cn = LU \ (B * Cn_1 + tau * F)
      Cn_1 = Cn
      Cns[n+1] = Cn
    end
  end
  return Cns
end

# Plots a gif with an approximate comparison to the exact function
function plot_comparisson(ne, tau, fr, alpha, beta, gamma, T, f, u, u0)
  # Initializes variables and arrays
  h = 1 / ne
  t = 0:tau:T
  N = length(t) - 1

  Cns = solve_system(ne, tau, alpha, beta, gamma, T, f, u0)

  # Iterating for plotting the approximate and exact function
  anim = @animate for n in 0:N
    Cn = Cns[n+1]
    plt = plot(label = "Aproximating u(x,t)", 
    xlabel = "x", size=(800, 800), xlim = (0, 1), ylims = (0, 0.15)) 
    plot!(plt, (x) -> u(x, t[n+1]), label = "Exact Function", color =:blue)
    plot!(plt, 0:h:1, [0 ; Cn; 0], label = "Approximation", linestyle =:dash, markershape=:circle, color =:red)
  end

  gif(anim, "plot-comparisson.gif", fps = fr)
end

# Computes the errors from a system of ne points and returns the maximum error
function error_from_system(ne, tau, alpha, beta, gamma, T, f, u, u0)
  # Initializing variables
  h = 1 / ne
  t = 0:tau:T
  N = length(t) - 1

  # Initializing matrices and vectors
  LG = init_LG_matrix(ne)
  EQ, m = init_EQ_vector_and_m(ne)
  errors = zeros(N+1)

  Cns = solve_system(ne, tau, alpha, beta, gamma, T, f, u0)

  for n in 0:N
    if (n == 0)
      errors[n+1] = gauss_error(u0, Cns[n+1], ne, EQ, LG)
    else
      errors[n+1] = gauss_error((x) -> u(x, t[n+1]), Cns[n+1], ne, EQ, LG)
    end
  end

  return maximum(errors)
end

# Plots the errors according to the variation of h = tau
function plot_error_convergence(lb, ub, alpha, beta, gamma, T, f, u, u0)
  vec_ne = [2^i for i in lb:ub]
  hs = 1 ./ vec_ne
  vec_tau = hs

  function error_convergence(lb, ub, vec_ne, vec_tau, alpha, beta, gamma, T, f, u, u0)
    errors = zeros(ub-lb+1)

    for i in lb:ub
      errors[i-lb+1] = error_from_system(vec_ne[i-lb+1], vec_tau[i-lb+1], alpha, beta, gamma, T, f, u, u0)
    end

    return errors
  end

  @time begin
    errors = error_convergence(lb, ub, vec_ne, vec_tau, alpha, beta, gamma, T, f, u, u0)
  end

  # Plots the errors in a log scale
  plot(hs, errors, seriestype = :scatter, label = "Error convergence", 
  xlabel = "h", ylabel = "error", size=(800, 800), xscale=:log10, yscale=:log10, 
  markercolor = :blue)
  plot!(hs, errors, seriestype = :line, label = "", linewidth = 2, linecolor = :blue)
  plot!(hs, hs.^2, seriestype = :line, label = "h^2", linewidth = 2, linecolor = :red)

  # Saves the graph in a png file
  savefig("error-convergence.png")

end


# Variables for plot_error_convergence
lb = 1    # lower-bound limit to 2^lb - 1
ub = 5    # upper-bound limit to 2^ub - 1

# Variables for plot_comparisson
ne = 5
tau = 1/8
frate = 5

# Constants
alpha  = 1
beta   = 1
gamma  = 0
T      = 1

# Functions
u  = (x,t) -> sin(π * x) * exp(-1 * t) / π^2
u0 = (x)   -> sin(π * x) / π^2
f  = (x,t) -> sin(π * x) * ((-1 + alpha * π^2 + beta) * exp(-1 * t) / π^2)


# Plots a png for analysing the error convergence
plot_error_convergence(lb, ub, alpha, beta, gamma, T, f, u, u0)

# Plots a gif for visualizing the results
plot_comparisson(ne, tau, frate, alpha, beta, gamma, T, f, u, u0)

