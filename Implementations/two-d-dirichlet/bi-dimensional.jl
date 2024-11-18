using Plots            # To use plot and gif
using GaussQuadrature  # To use legendre
using SparseArrays     # To use spzeros
using LinearAlgebra    # To use lu


# Approximates the Integral of a given function by the gaussian quadrature technic
function gauss_quad(f, ngp)

  # Initializes P and W according to the number of Gauss points
  P, W = legendre(ngp)
  sum = 0

  # Calculates the sum
  for j in 1:ngp
    sum = sum + W[j] * f(P[j])
  end

  return sum
end

# Approximates the Integral of a given function by the gaussian quadrature technic
function double_gauss_quad(f, ngp)

  # Initializes P and W according to the number of Gauss points
  P, W = legendre(ngp)
  sum = 0

  # Calculates the sum
  for i in 1:ngp
    for j in 1:ngp
      sum = sum + W[j] * W[i] * f(P[j], P[i])
    end
  end

  return sum
end

# Phi function-vector
phi = [(xi1, xi2) -> (1 - xi1) * (1 - xi2) * (1 / 4);
       (xi1, xi2) -> (1 + xi1) * (1 - xi2) * (1 / 4);
       (xi1, xi2) -> (1 + xi1) * (1 + xi2) * (1 / 4);
       (xi1, xi2) -> (1 - xi1) * (1 + xi2) * (1 / 4)
]

# Derivative of the phi function-vector
d_qsi_phi = [
  [((xi1, xi2) -> (-1 / 4) * (1 - xi2)),
   ((xi1, xi2) -> ( 1 / 4) * (1 - xi2)),
   ((xi1, xi2) -> ( 1 / 4) * (1 + xi2)),
   ((xi1, xi2) -> (-1 / 4) * (1 + xi2))],
  [((xi1, xi2) -> (-1 / 4) * (1 - xi1)),
   ((xi1, xi2) -> (-1 / 4) * (1 + xi1)),
   ((xi1, xi2) -> ( 1 / 4) * (1 + xi1)),
   ((xi1, xi2) -> ( 1 / 4) * (1 - xi1))]
]

# Converts the interval
x = [
  (xi, eta, h1, h2, pe1, pe2) -> (h1 / 2) * (xi + 1) + pe1;
  (xi, eta, h1, h2, pe1, pe2) -> (h2 / 2) * (eta + 1) + pe2
]

# Initializes the LG matrix
function init_LG_matrix(Nx1, Nx2)
  ne = Nx1 * Nx2
  LG = fill(0, (4, ne))
  j = 0

  for i in 1:ne
    if (j % (Nx1+1) == 0)
      j = j + 1
    end

    LG[1, i] = j
    LG[2, i] = j + 1
    LG[3, i] = j + Nx1 + 2
    LG[4, i] = j + Nx1 + 1

    j = j + 1
  end

  return LG
end

# Initializes the EQ Vector
function init_EQ_vector_and_m(Nx1, Nx2)
  m = (Nx1 - 1) * (Nx2 -1)
  EQ = fill(0, (Nx2+1, Nx1+1))

  # Initializes the border elements
  for i in 1:Nx1+1
    EQ[1,i] = m + 1
    EQ[Nx2+1, i] = m + 1
  end
  for j in 1:Nx2+1
    EQ[j, 1] = m+1
    EQ[j, Nx1+1] = m+1
  end

  # initializes the within elements
  k = 1
  for i in 2:Nx2
    for j in 2:Nx1
      EQ[i,j] = k
      k = k + 1
    end
  end

  return cat(EQ'..., dims=1), m
end

# Initializes the K matrix
function init_K_matrix(alpha, beta, Nx1, Nx2, m, EQ, LG)

  # Initializes the Ke matrix
  function init_Ke_matrix(alpha, beta, Nx1, Nx2)
    Ke = zeros(4,4)
    h1 = 1 / Nx1
    h2 = 1 / Nx2
    ngp = 2

    for a in 1:4
      for b in 1:4
        Ke[a,b] = (alpha * h2 / h1)    * double_gauss_quad((xi1, xi2) -> d_qsi_phi[1][b](xi1, xi2) * d_qsi_phi[1][a](xi1, xi2), ngp) +
                  (alpha * h1 / h2)    * double_gauss_quad((xi1, xi2) -> d_qsi_phi[2][b](xi1, xi2) * d_qsi_phi[2][a](xi1, xi2), ngp) +
                  (beta * h1 * h2 / 4) * double_gauss_quad((xi1, xi2) ->       phi[b](xi1, xi2)    *       phi[a](xi1, xi2),    ngp)
      end
    end

    return Ke
  end

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

  # Initializes the Fe vector
  function init_Fe_vector(f, Nx1, Nx2, pe1, pe2)
    Fe = zeros(4)
    h1 = 1 / Nx1
    h2 = 1 / Nx2
    ngp = 5
    M =

    for a in 1:4
      Fe[a] = h1 * h2 / 4 * double_gauss_quad((xi1, xi2) -> f(x[1](xi1, xi2, h1, h2, pe1, pe2),
                                                              x[2](xi1, xi2, h1, h2, pe1, pe2)) *
                                                            phi[a](xi1, xi2), ngp)
    end

    return Fe
  end

  F = zeros(m+1)

  for j in 1:Nx2
    pe2 = (j - 1) * (1 / Nx2)
    for i in 1:Nx1
      pe1 = (i - 1) * (1 / Nx1)
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
function solve_system(alpha, beta, Nx1, Nx2, f; EQLG=false)
  EQ, m = init_EQ_vector_and_m(Nx1, Nx2)
  LG = init_LG_matrix(Nx1, Nx2)
  K = init_K_matrix(alpha, beta, Nx1, Nx2, m, EQ, LG)
  F = init_F_vector(f, Nx1, Nx2, m, EQ, LG)
  C = K \ F
  return EQLG ? (C, EQ, LG) : C
end

# Plots the approximation found
function plot_approximation(alpha, beta, Nx1, Nx2, f)

  # Initializes the axes and computes the temperatures (including the boundary condition) (works only in linear bases)
  C_in_plane = vcat(zeros(1, Nx2+1), hcat(zeros(Nx1-1, 1), reshape(solve_system(alpha, beta, Nx1, Nx2, f), Nx1-1, Nx2-1), zeros(Nx1-1, 1)), zeros(1, Nx2+1))'
  ext_xs1 = [i * 1/Nx1 for i in 0:Nx1]
  ext_xs2 = [i * 1/Nx2 for i in 0:Nx2]

  # Plots the approximation
  #heatmap(C_in_plane, color=:thermal, title="approximation")
  wireframe(ext_xs1, ext_xs2, C_in_plane, linecolor=:black, lw=1, n=5, size=(500,500))
  surface!(ext_xs1, ext_xs2, C_in_plane, color=:thermal, alpha=0.5, title="Approximation found for u(x1, x2)", xlabel="x1", ylabel="x2", zlabel="Temperatures", n=5)

  savefig("approximation_found.png")
end

# Plots the error converge
function plot_error_convergence(lb, ub, alpha, beta, u, f)

  # Computes the error of a system given C
  function gauss_error(u, C, Nx1, Nx2, EQ, LG)
    h1 = 1 / Nx1
    h2 = 1 / Nx2
    C_ext = [C ; 0]
    erro = 0

    for j in 1:Nx2
      p2 = (j - 1) * h2

      for i in 1:Nx1
        p1 = (i - 1) * h2
        e = (j - 1) * Nx1 + i

        function integrand(xi1, xi2, e)
          x1 = x[1](xi1, xi2, h1, h2, p1, p2)
          x2 = x[2](xi1, xi2, h1, h2, p1, p2)
          aprox = C_ext[EQ[LG[1, e]]] * phi[1](xi1, xi2) +
                  C_ext[EQ[LG[2, e]]] * phi[2](xi1, xi2) +
                  C_ext[EQ[LG[3, e]]] * phi[3](xi1, xi2) +
                  C_ext[EQ[LG[4, e]]] * phi[4](xi1, xi2)
          return (u(x1, x2) - aprox)^2
        end

        erro += double_gauss_quad((xi1, xi2) -> integrand(xi1, xi2, e), 5)
      end
    end

    return sqrt(erro * h1 * h2 / 4)
  end

  Nxs = [2^i for i in lb:ub]
  errors = zeros(ub - lb + 1)

  for i in eachindex(Nxs)
    C, EQ, LG = solve_system(alpha, beta, Nxs[i], Nxs[i], f; EQLG=true)
    errors[i] = gauss_error(u, C, Nxs[i], Nxs[i], EQ, LG)
  end

    # Plots the errors in a log scale
    hs = 1 ./ Nxs
    plot(hs, errors, seriestype = :scatter, label = "Error convergence",
    xlabel = "h", ylabel = "error", size=(800, 800), xscale=:log10, yscale=:log10,
    markercolor = :blue)
    plot!(hs, errors, seriestype = :line, label = "", linewidth = 2, linecolor = :blue)
    plot!(hs, hs.^2, seriestype = :line, label = "h^2", linewidth = 2, linecolor = :red)

    # Saves the graph in a png file
    savefig("error-convergence.png")

end

# Plots the comparison between our approximation and the exact function
function plot_comparison(alpha, beta, Nx1, Nx2, f, u)
  # Initializes the axes and computes the temperatures (including the boundary condition) (works only in linear bases)
  C_in_plane = vcat(zeros(1, Nx2+1), hcat(zeros(Nx1-1, 1), reshape(solve_system(alpha, beta, Nx1, Nx2, f), Nx1-1, Nx2-1), zeros(Nx1-1, 1)), zeros(1, Nx2+1))'
  ext_xs1 = [i * 1/Nx1 for i in 0:Nx1]
  ext_xs2 = [i * 1/Nx2 for i in 0:Nx2]
  U_in_plane = u.(ext_xs1', ext_xs2)

  # Plots the approximation
  wireframe(ext_xs1, ext_xs2, C_in_plane, linecolor=:black, lw=1, n=5, size=(500,500))
  surface!(ext_xs1, ext_xs2, C_in_plane, color=:thermal, alpha=0.5, title="Approximation found for u(x1, x2)", xlabel="x1", ylabel="x2", zlabel="Temperatures", n=5)
  surface!(ext_xs1, ext_xs2, U_in_plane, color=:blues, alpha=0.5, n=5)
  savefig("comparison.png")
end

# Constants
alpha = 1
beta = 1

# Discretization for plot_approximation and plot_comparison
Nx1 = 3
Nx2 = 100

# Functions
u     = (x1,x2) -> sin(pi * x1) * sin(pi * x2)
ux1x1 = (x1,x2) -> -1 * pi^2 * sin(pi * x1) * sin(pi * x2)
ux2x2 = (x1,x2) -> -1 * pi^2 * sin(pi * x1) * sin(pi * x2)
f     = (x1,x2) -> (-1 * alpha * ux1x1(x1,x2)) + (-1 * alpha * ux2x2(x1,x2)) + beta * u(x1,x2)

# Functions call for testing the implementation
plot_approximation(alpha, beta, Nx1, Nx2, f)
plot_comparison(alpha, beta, Nx1, Nx2, f, u)
plot_error_convergence(2, 6, alpha, beta, u, f)
