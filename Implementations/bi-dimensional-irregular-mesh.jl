using Plots            # To use plot and gif
using GaussQuadrature  # To use legendre
using SparseArrays     # To use spzeros
using LinearAlgebra    # To use lu
using BenchmarkTools

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
phi = [
  (xi1, xi2) -> (1 - xi1) * (1 - xi2) * (1 / 4);
  (xi1, xi2) -> (1 + xi1) * (1 - xi2) * (1 / 4);
  (xi1, xi2) -> (1 + xi1) * (1 + xi2) * (1 / 4);
  (xi1, xi2) -> (1 - xi1) * (1 + xi2) * (1 / 4)
]

# Derivative of the phi function-vector
d_xi_phi = [
  [((xi1, xi2) -> (-1 / 4) * (1 - xi2)),
   ((xi1, xi2) -> ( 1 / 4) * (1 - xi2)),
   ((xi1, xi2) -> ( 1 / 4) * (1 + xi2)),
   ((xi1, xi2) -> (-1 / 4) * (1 + xi2))],
  [((xi1, xi2) -> (-1 / 4) * (1 - xi1)),
   ((xi1, xi2) -> (-1 / 4) * (1 + xi1)),
   ((xi1, xi2) -> ( 1 / 4) * (1 + xi1)),
   ((xi1, xi2) -> ( 1 / 4) * (1 - xi1))]
]

x = [
  (xi1, xi2, Xs) -> dot(Xs, map((f) -> f(xi1, xi2), phi)),
  (xi1, xi2, Ys) -> dot(Ys, map((f) -> f(xi1, xi2), phi))
]

d_xi_x = [
  [(xi1, xi2, Xs) -> dot(Xs, map((f) -> f(xi1, xi2), d_xi_phi[1])),
   (xi1, xi2, Xs) -> dot(Xs, map((f) -> f(xi1, xi2), d_xi_phi[2]))],
  [(xi1, xi2, Ys) -> dot(Ys, map((f) -> f(xi1, xi2), d_xi_phi[1])),
   (xi1, xi2, Ys) -> dot(Ys, map((f) -> f(xi1, xi2), d_xi_phi[2]))]
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
  m = (Nx1 - 1) * (Nx2 - 1)
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

# Initializes the Ke matrix
function init_Ke_matrix(alpha, beta, Xs, Ys)
  Ke = zeros(4,4)
  ngp = 5
  P, W = legendre(ngp)

  J = (xi1, xi2) -> abs(d_xi_x[1][1](xi1, xi2, Xs) * d_xi_x[2][2](xi1, xi2, Ys) -
                        d_xi_x[1][2](xi1, xi2, Xs) * d_xi_x[2][1](xi1, xi2, Ys))

  H = (xi1, xi2) -> [
      d_xi_x[2][2](xi1, xi2, Ys)   -d_xi_x[2][1](xi1, xi2, Ys);
    - d_xi_x[1][2](xi1, xi2, Xs)    d_xi_x[1][1](xi1, xi2, Xs)]

  for i in 1:ngp
    for j in 1:ngp
      xi1 = P[j]
      xi2 = P[i]
      det_J = J(xi1, xi2)
      HTH = (H(xi1, xi2)' * H(xi1, xi2))


      for a in 1:4
        d_xi_phi_1a = d_xi_phi[1][a]
        d_xi_phi_2a = d_xi_phi[2][a]
        for b in 1:4
          Ke[a, b] += W[j] * W[i] * (alpha * ([d_xi_phi[1][b](xi1, xi2);
                                              d_xi_phi[2][b](xi1, xi2)]' *
                                             HTH *
                                             [d_xi_phi_1a(xi1, xi2);
                                              d_xi_phi_2a(xi1, xi2)] *
                                             (1 / det_J)) +
                                    beta * (phi[b](xi1, xi2) *
                                            phi[a](xi1, xi2) *
                                            det_J))
        end
      end
    end
  end

  return Ke
end

# Initializes the K matrix
function init_K_matrix(alpha, beta, X_matrix, Y_matrix, m, EQ, LG)
  ne = size(LG, 2) # assuming we are using a LG (4 x ne)
  K = spzeros(m+1, m+1)

  for e in 1:ne
    Ke = init_Ke_matrix(alpha, beta, X_matrix[LG[:,e]], Y_matrix[LG[:,e]])
    for b in 1:4
      j = EQ[LG[b, e]]
      for a in 1:4
        i = EQ[LG[a, e]]
        K[i,j] += Ke[a,b]
      end
    end
  end

  return K[1:m, 1:m]
end

# Initializes the Fe vector
function init_Fe_vector(f, Xs, Ys)
  Fe = zeros(4)
  ngp = 5

  for a in 1:4
    Fe[a] = double_gauss_quad((xi1, xi2) -> f(x[1](xi1, xi2, Xs), x[2](xi1, xi2, Ys)) *
                                            phi[a](xi1, xi2) *
                                            abs(d_xi_x[1][1](xi1, xi2, Xs) *
                                                d_xi_x[2][2](xi1, xi2, Ys) -
                                                d_xi_x[1][2](xi1, xi2, Xs) *
                                                d_xi_x[2][1](xi1, xi2, Ys)), ngp)
  end

  return Fe
end

# Initializes the F vector
function init_F_vector(f, X_matrix, Y_matrix, m, EQ, LG)
  ne = size(LG, 2) # assuming we are using a LG (4 x ne)
  F = zeros(m+1)

  for e in 1:ne
    Fe = init_Fe_vector(f, X_matrix[LG[:,e]], Y_matrix[LG[:,e]])
    for a in 1:4
      F[EQ[LG[a,e]]] += Fe[a]
    end
  end

  return F[1: end-1]
end

# Solves the system using a regular mesh
function solve_system(alpha, beta, f, Nx1, Nx2; EQLG=false, XY_matrix=false, noise=false)
  X_matrix, Y_matrix = init_mesh(Nx1, Nx2, ns=noise)
  EQ, m = init_EQ_vector_and_m(Nx1, Nx2)
  LG = init_LG_matrix(Nx1, Nx2)
  K = init_K_matrix(alpha, beta, X_matrix, Y_matrix, m, EQ, LG)
  F = init_F_vector(f, X_matrix, Y_matrix, m, EQ, LG)
  C = K \ F
  return EQLG ? XY_matrix ? (C, EQ, LG, X_matrix, Y_matrix) : (C, EQ, LG) : C
end

# Plots the approximation for a linear base
function plot_approximation(alpha, beta, f, Nx1, Nx2)

  # Initializes the axes and computes the temperatures (incluiding the boundary condition) (works only in linear bases)
  C_in_plane = vcat(zeros(1, Nx2+1), hcat(zeros(Nx1-1, 1), reshape(solve_system(alpha, beta, f, Nx1, Nx2; EQLG=false), Nx1-1, Nx2-1), zeros(Nx1-1, 1)), zeros(1, Nx2+1))'
  ext_xs1 = [i * 1/Nx1 for i in 0:Nx1]
  ext_xs2 = [i * 1/Nx2 for i in 0:Nx2]

  # Plots the approximation
  wireframe(ext_xs1, ext_xs2, C_in_plane, linecolor=:black, lw=1, n=5, size=(500,500))
  surface!(ext_xs1, ext_xs2, C_in_plane, color=:thermal, alpha=0.5, title="Approximation found for u(x1, x2)", xlabel="x1", ylabel="x2", zlabel="Temperatures", n=5)

  savefig("approximation_found.png")
end

# Plots the error converge
function error_convergence(lb, ub, alpha, beta, u, f; see_plot=false, ns=false)

  # Computes the error of a system given C
  function gauss_error(u, C, X_matrix, Y_matrix, EQ, LG)
    C_ext = [C ; 0]
    erro = 0
    ne = size(LG, 2) # assuming we are using a LG (4 x ne)

    function integrand(xi1, xi2, Xs, Ys, e)
      x1 = x[1](xi1, xi2, Xs)
      x2 = x[2](xi1, xi2, Ys)
      aprox = C_ext[EQ[LG[1, e]]] * phi[1](xi1, xi2) +
              C_ext[EQ[LG[2, e]]] * phi[2](xi1, xi2) +
              C_ext[EQ[LG[3, e]]] * phi[3](xi1, xi2) +
              C_ext[EQ[LG[4, e]]] * phi[4](xi1, xi2)
      J_det = d_xi_x[1][1](xi1, xi2, Xs) * d_xi_x[2][2](xi1, xi2, Ys) - d_xi_x[1][2](xi1, xi2, Xs) * d_xi_x[2][1](xi1, xi2, Ys)
      return (u(x1, x2) - aprox)^2 * J_det
    end

    for e in 1:ne
      er = double_gauss_quad((xi1, xi2) -> integrand(xi1, xi2, X_matrix[LG[:,e]], Y_matrix[LG[:,e]], e), 5)
      erro += er
    end

    return sqrt(erro)
  end

  Nxs = [2^i for i in lb:ub]
  errors = zeros(ub - lb + 1)

  for i in lb:ub
    display(i)
    C, EQ, LG, X_matrix, Y_matrix = solve_system(alpha, beta, f, Nxs[i-lb+1], Nxs[i-lb+1]; EQLG=true, XY_matrix=true, noise=ns)
    errors[i-lb+1] = gauss_error(u, C, X_matrix , Y_matrix, EQ, LG)
  end

  if see_plot
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
end

# Initializes Xs and Ys as a regular mesh
function init_mesh(Nx1, Nx2; ns=false)
  h1 = 1 / Nx1
  h2 = 1 / Nx2

  x1 = collect(0:h1:1)
  x2 = collect(0:h2:1)

  X = [x1[i] for i in 1:Nx1+1, j in 1:Nx2+1]
  Y = [x2[j] for i in 1:Nx1+1, j in 1:Nx2+1]

  if ns == true
    @assert size(X) == size(Y) "X e Y must have the same dimensions"
    if size(X, 1) > 2 && size(X, 2) > 2
      rl1, rl2 = h1 / 4, h2 / 4
      X[2:end-1, 2:end-1] .+= rl1 * (rand(Float64, size(X[2:end-1,2:end-1])) .- 0.5) * 2
      Y[2:end-1, 2:end-1] .+= rl2 * (rand(Float64, size(Y[2:end-1,2:end-1])) .- 0.5) * 2
    end
  end

  return X, Y
end

# Plots the mesh
function plot_mesh(X, Y, LG)
  fig = Plots.plot(legend=false, aspect_ratio=:equal, xticks=0:0.25:1, yticks=0:0.25:1)

  Plots.scatter!(X, Y, markersize=4, color=:blue)

  for e in 1:size(LG, 2)
    i1, i2, i3, i4 = LG[:, e]

  Plots.plot!([X[i1], X[i2], X[i3], X[i4], X[i1]],
    [Y[i1], Y[i2], Y[i3], Y[i4], Y[i1]], color=:black)
  end

  savefig("malha2d-teste.png")
end

# Constants
alpha = 1
beta = 1

# Discretization for plot_approximation and plot_comparisson
Nx1 = 5
Nx2 = 5

# Functions
u     = (x1,x2) -> sin(pi * x1) * sin(pi * x2)
ux1x1 = (x1,x2) -> -1 * pi^2 * sin(pi * x1) * sin(pi * x2)
ux2x2 = (x1,x2) -> -1 * pi^2 * sin(pi * x1) * sin(pi * x2)
f     = (x1,x2) -> (-1 * alpha * ux1x1(x1,x2)) + (-1 * alpha * ux2x2(x1,x2)) + beta * u(x1,x2)


# Functions call for testing the implementation
plot_approximation(alpha, beta, f, Nx1, Nx2)
error_convergence(2, 5, alpha, beta, u, f, see_plot=true, ns=true)
LG = init_LG_matrix(Nx1, Nx2)
X, Y = init_mesh(Nx1, Nx2, ns=true)
plot_mesh(X, Y, LG)

# t2 = @time plot_error_convergence(2, 7, alpha, beta, u, f)
    #  best_time2 = minimum(t2.times)
    #  b = best_time2 / 1e9  # Converting to seconds

    # display(b)

