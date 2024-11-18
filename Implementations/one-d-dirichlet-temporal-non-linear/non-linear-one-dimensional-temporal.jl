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

# Initializes the K matrix
function init_K_matrix(ne, EQ, LG, alpha, beta, gamma, m)

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
function init_F_vector(f, ne, EQ, LG, m)

  # Initializes the Fe vector
  function init_Fe_vector(f, ne, e)
    Fe = zeros(2)
    h = 1 / ne

    for a in 1:2
      Fe[a] = (h / 2) * gaussian_quadrature((qsi) -> f(qsi_to_x(qsi, e, h)) *  phi(a, qsi), 5)
    end

    return Fe
  end

  F = zeros(m+1)

  for e in 1:ne
    Fe = init_Fe_vector(f, ne, e)
    for a in 1:2
      F[EQ[LG[a,e]]] += Fe[a]
    end
  end

  # Removes the last line
  return F[1:m]
end

# Initializes the C0 third option vector
function init_C0_3rd_option_vector(u0x, ne, EQ, LG, m)

  # Initializes the C0e vector
  function init_C0e_3rd_option_vector(u0x, ne, e)
    C0e = zeros(2)
    h = 1 / ne

    for a in 1:2
      C0e[a] = gaussian_quadrature((qsi) -> u0x(qsi_to_x(qsi, e, h)) *  d_phi(a, qsi), 5)
    end

    return C0e
  end

  C0 = zeros(m+1)

  for e in 1:ne
    C0e = init_C0e_3rd_option_vector(u0x, ne, e)
    for a in 1:2
      C0[EQ[LG[a,e]]] += C0e[a]
    end
  end

  # Removes the last line
  return C0[1:m]
end

# Initializes the C0 vector
function init_C0_vector(option, u0, u0x, ne, EQ, LG, alpha, beta, gamma, m)
  if (option == 1)
    h = 1 / ne
    return u0.(h:h:1-h)

  elseif (option == 2)
    M = init_K_matrix(ne, EQ, LG, 0, 1, 0, m)
    b = init_F_vector(u0, ne, EQ, LG, m)
    return M \ b

  elseif (option == 3)
    A = init_K_matrix(ne, EQ, LG, 1, 0, 0, m)
    b = init_C0_3rd_option_vector(u0x, ne, EQ, LG, m)
    return A \ b

  elseif (option == 4)
    h = 1 / ne
    K = init_K_matrix(ne, EQ, LG, alpha, beta, gamma, m)
    b = (alpha .* init_C0_3rd_option_vector(u0x, ne, EQ, LG, m)) + (beta .* init_F_vector(u0, ne, EQ, LG, m)) + (gamma .* init_F_vector(u0x, ne, EQ, LG, m))
    return K \ b

  else
    error("option not recognized in init_C0_vector")
  end
end

# Initializes the G vector
function init_G_vector(g, ne, EQ, LG, C, m, h)

  # Initializes the Ge vector
  function init_Ge_vector(g, e, EQ, LG, C, h)
    Ge = zeros(2)

    for a in 1:2
      Ge[a] = h / 2 * gaussian_quadrature((qsi) -> g(C[EQ[LG[1, e]]] * phi(1, qsi) + C[EQ[LG[2, e]]] * phi(2, qsi)) * phi(a, qsi), 5)
    end

    return Ge
  end

  G = zeros(m+1)
  ext_C = [C ; 0]

  for e in 1:ne
    Ge = init_Ge_vector(g, e, EQ, LG, ext_C, h)
    for a in 1:2
      G[EQ[LG[a,e]]] += Ge[a]
    end
  end

  # Removes the last line
  return G[1:m]
end

# Generalized phi function
function phi(num, qsi)
  return [((1 - qsi) / 2), ((1 + qsi) / 2)][num]
end

# Generalized derivative of the phi function
function d_phi(num, qsi)
  return [(-1 / 2), (1 / 2)][num]
end

# Converts the interval from [x_i-1 , xi+1] to [-1, 1]
function qsi_to_x(qsi, i, h)
  return (h / 2) * (qsi + 1) + 0 + (i - 1)*h
end

# Solves the system and returns Cns
function solve_system(ne, tau, alpha, beta, gamma, T, f, u0, u0x, g, C0option)
  # Initializing variables, matrices and vectors
  h = 1 / ne
  t = 0:tau:T
  N = length(t) - 1
  LG = init_LG_matrix(ne)
  EQ, m = init_EQ_vector_and_m(ne)
  K = init_K_matrix(ne, EQ, LG, alpha, beta, gamma, m)
  M = init_K_matrix(ne, EQ, LG, 0, 1, 0, m)
  A = M + ((tau / 2) * K)
  B = M - ((tau / 2) * K)
  LU = lu(A)
  Cn = zeros(m)
  Cn_1 = zeros(m)
  Cn_2 = zeros(m)
  Cns = Vector{Vector{Float64}}(undef, N+1)

  # Iterations for computing u by varying n (time)
  for n in 0:N
    if (n == 0)
      Cn = init_C0_vector(C0option, u0, u0x, ne, EQ, LG, alpha, beta, gamma, m)

      # Setting the vectors up for the remaining iterations and storing CN
      Cns[n+1] = Cn
      Cn_2 = Cn_1
      Cn_1 = Cn
    elseif (n == 1)
      # First part
      G = init_G_vector(g, ne, EQ, LG , Cn_1, m, h)
      F = init_F_vector((x) -> f(x, (n*tau - tau/2)), ne, EQ, LG, m)
      C_til = LU \ (B * Cn_1 - tau * G + tau * F)

      # Second part
      G = init_G_vector(g, ne, EQ, LG , (Cn_1 + C_til) / 2, m, h)
      Cn = LU \ (B * Cn_1 - tau * G + tau * F)

      # Setting the vectors up for the remaining iterations and storing CN
      Cns[n+1] = Cn
      Cn_2 = Cn_1
      Cn_1 = Cn
    else
      F = init_F_vector((x) -> f(x, (n*tau - tau/2)), ne, EQ, LG, m)
      G = init_G_vector(g, ne, EQ, LG, (3 * Cn_1 - Cn_2) / 2, m, h)
      Cn = LU \ (B * Cn_1 - tau * G + tau * F )

      # Setting the vectors up for the remaining iterations and storing CN
      Cns[n+1] = Cn
      Cn_2 = Cn_1
      Cn_1 = Cn
    end
  end
  return Cns
end

# Plots a gif with an approximate comparison to the exact function
function plot_comparison(ne, tau, fr, alpha, beta, gamma, T, f, u, u0, u0x, g, C0option)

  # Initializes variables and arrays
  h = 1 / ne
  t = 0:tau:T
  N = length(t) - 1

  Cns = solve_system(ne, tau, alpha, beta, gamma, T, f, u0, u0x, g, C0option)

  # Iterating for plotting the approximate and exact function
  anim = @animate for n in 0:N
    Cn = Cns[n+1]
    plt = plot(label = "Approximating u(x,t)",
    xlabel = "x", size=(800, 800), xlim = (0, 1), ylims = (0, 0.15))
    plot!(plt, (x) -> u(x, t[n+1]), label = "Exact Function", color =:blue)
    plot!(plt, 0:h:1, [0 ; Cn; 0], label = "Approximation", linestyle =:dash, markershape=:circle, color =:red)
  end

  gif(anim, "plot-comparison.gif", fps = fr)
end

# Computes the errors from a system of ne points and returns the maximum error
function error_from_system(ne, tau, alpha, beta, gamma, T, f, u, u0, u0x, g, C0option)

  # Computes the error according to ne
  function gauss_error(u, cs, ne, EQ, LG)
    sum = 0
    h = 1 / ne

    # Including 0 so that the EQ-LG will not consider the first and the last phi function
    ext_cs = [cs ; 0]

    # Computing the error
    for e in 1:ne
      sum = sum + gaussian_quadrature((qsi) -> (u(qsi_to_x(qsi, e, h)) - (ext_cs[EQ[LG[1,e]]] * phi(1, qsi)) - (ext_cs[EQ[LG[2,e]]] * phi(2, qsi)))^2, 5)
    end

    return sqrt(sum * (h / 2))
  end

  # Initializing variables
  h = 1 / ne
  t = 0:tau:T
  N = length(t) - 1

  # Initializing matrices and vectors
  LG = init_LG_matrix(ne)
  EQ, m = init_EQ_vector_and_m(ne)
  errors = zeros(N+1)

  Cns = solve_system(ne, tau, alpha, beta, gamma, T, f, u0, u0x, g, C0option)

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
function plot_error_convergence(lb, ub, alpha, beta, gamma, T, f, u, u0, u0x, g, C0option)
  vec_ne = [2^i for i in lb:ub]
  hs = 1 ./ vec_ne
  vec_tau = hs

  function error_convergence(lb, ub, vec_ne, vec_tau, alpha, beta, gamma, T, f, u, u0, u0x, g, C0option)
    errors = zeros(ub-lb+1)

    for i in lb:ub
      errors[i-lb+1] = error_from_system(vec_ne[i-lb+1], vec_tau[i-lb+1], alpha, beta, gamma, T, f, u, u0, u0x, g, C0option)
    end

    return errors
  end

  errors = error_convergence(lb, ub, vec_ne, vec_tau, alpha, beta, gamma, T, f, u, u0, u0x, g, C0option)

  # Plots the errors in a log scale
  plot(hs, errors, seriestype = :scatter, label = "Error convergence",
  xlabel = "h", ylabel = "error", size=(800, 800), xscale=:log10, yscale=:log10,
  markercolor = :blue)
  plot!(hs, errors, seriestype = :line, label = "", linewidth = 2, linecolor = :blue)
  plot!(hs, hs.^2, seriestype = :line, label = "h^2", linewidth = 2, linecolor = :red)

  # Saves the graph in a png file
  savefig("error-convergence-$(C0option)-c0option.png")

end

# Plots the approximation function
function plot_approximation(ne, tau, fr, alpha, beta, gamma, T, f, u0, u0x, g, C0option)
  Cns = solve_system(ne, tau, alpha, beta, gamma, T, f, u0, u0x, g, C0option)
  h = 1 / ne

  # Iterating for plotting the approximate function
  anim = @animate for n in 1:length(Cns)
    Cn = Cns[n]
    plt = plot(label = "Approximating u(x,t)",
    xlabel = "x", size=(800, 800), xlim = (0, 1), ylims = (0, 0.15))
    plot!(plt, 0:h:1, [0 ; Cn; 0], label = "Approximation", linestyle =:dash, markershape=:circle, color =:red)
  end

  gif(anim, "plot-approximation.gif", fps = fr)
end



# Variables for plot_error_convergence
lb = 1    # lower-bound limit to 2^lb - 1
ub = 5    # upper-bound limit to 2^ub - 1

# Variables for plot_comparison
ne = 4
tau = 1/8
frate = 4

# Constants
alpha  = 1
beta   = 1
gamma  = 0
T      = 1

# Functions
# u   = (x,t) -> sin(π * x) * exp(-1 * t) / π^2
# u0  = (x)   -> sin(π * x) / π^2
# u0x = (x)   -> cos(π * x) / π
# f   = (x,t) -> sin(π * x) * ((-1 + alpha * π^2 + beta) * exp(-1 * t) / π^2) + g(u(x,t))
# g   = (s)   -> s^3 - 2 * s

u   = (x,t) -> sin(π * x) * exp(-1 * t) / π^2
u0  = (x)   -> sin(π * x) / π^2
u0x = (x)   -> cos(π * x) / π
ux  = (x,t) -> cos(π * x) * exp(-1 * t) / π
uxx = (x,t) -> sin(π * x) * -1 * exp(-1 * t)
ut  = (x,t) -> sin(π * x) * -1 * exp(-1 * t) / π^2
g   = (s)   -> s^3 - 2 * s
f   = (x,t) -> ut(x,t) + (-1 * alpha * uxx(x,t)) + (gamma * ux(x,t)) + (beta * u(x,t)) + g(u(x,t))

# Saves a png for analyzing the error convergence
plot_error_convergence(lb, ub, alpha, beta, gamma, T, f, u, u0, u0x, g, 1)
plot_error_convergence(lb, ub, alpha, beta, gamma, T, f, u, u0, u0x, g, 2)
plot_error_convergence(lb, ub, alpha, beta, gamma, T, f, u, u0, u0x, g, 3)
plot_error_convergence(lb, ub, alpha, beta, gamma, T, f, u, u0, u0x, g, 4)

# Saves a gif for visualizing the results
plot_comparison(ne, tau, frate, alpha, beta, gamma, T, f, u, u0, u0x, g, 1)

# Saves a png for visualizing the approximation
plot_approximation(ne, tau, frate, alpha, beta, gamma, T, (x, t) -> 0, u0, u0x, g, 1)