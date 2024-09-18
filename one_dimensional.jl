using Plots            # To use plot
using GaussQuadrature  # To use legendre
using SparseArrays     # To use spzeros


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
  LG = zeros(Int, 2,ne)

    for j in 1:ne
      LG[1,j] = j
      LG[2,j] = j + 1
    end

  return LG
end


# Initializes the EQ vector
function init_EQ_vector(ne)
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

  return EQ
end


# Initializes the Ke matrix
function init_Ke_matrix(ne)
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
function init_K_matrix(ne)
  K = spzeros(ne,ne)
  h = 1 / ne
  m = ne - 1
  Ke = init_Ke_matrix(ne)

  # Computes the EQ, LG and EQoLG matrices and vectors
  EQ = init_EQ_vector(ne)
  LG = init_LG_matrix(ne)

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
function init_Fe_vector(ne, e)
  Fe = zeros(2)
  h = 1 / ne

  for a in 1:2
    Fe[a] = (h / 2) * gaussian_quadrature((qsi) -> f(qsi_to_x(qsi, e, h)) *  phi(a, qsi), 5)
  end

  return Fe
end


# Initializes the F vector
function init_F_vector(ne)
  h = 1 / ne
  F = zeros(ne)
  m = ne - 1

  # Computes the EQ, LG and EQoLG matrices and vectors
  EQ = init_EQ_vector(ne)
  LG = init_LG_matrix(ne)
  EQoLG = EQ[LG]

  for e in 1:ne
    Fe = init_Fe_vector(ne, e)
    for a in 1:2
      i = EQoLG[a,e]
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
  return (h / 2) * (qsi + 1) + x0 + (i - 1)*h 
end


# Computes the error according to ne
function calc_error(cs, ne)
  sum = 0
  h = 1 / ne
  extended_cs = [cs; 0]
  EQ = init_EQ_vector(ne)
  LG = init_LG_matrix(ne)

  # Computes the error 
  for e in 1:ne
    sum = sum + gaussian_quadrature((qsi) -> (u(qsi_to_x(qsi, e, h)) - (extended_cs[EQ[LG[1,e]]] * phi(1, qsi)) - (extended_cs[EQ[LG[2,e]]] * phi(2, qsi)))^2, 5)
  end

  return sqrt(sum * (h / 2))
end


# Plots the exact and inexact graphs, as well as the absolute and relative errors
function plot_comparisson(ne)

  # Initializes h
  h = 1 / ne

  # Defines xs
  xs = zeros(ne - 1)

  # Initializes xs
  for i in 1:ne-1
    xs[i] = h * i
  end

  # Solves the system
  K = init_K_matrix(ne)
  F = init_F_vector(ne)
  cs = K \ F

  # Includes the boundary conditions in xs and uhs
  xs = [0; xs; 1]
  uhs = [0; cs; 0]

  # Getting the plot with the exact function
  plt = plot(u, 0, 1, label = "Expected Function", size=(800, 800))

  # Plotting the exact function with our approximations
  plot!(plt, xs, uhs, seriestype = :scatter, label = "Galerkin approximation", xlabel = "x", ylabel = "Cs 'u(x)' ", size=(800, 800))

  # Saving the graph
  savefig("Galerkin-generic-ode-error-graph.png")

  # Initializing errors lists
  abs_errors = zeros(ne + 1)

  # Calculating the absolute errors
  for i in 1:ne+1
    xi = xs[i]
    fxi = u(xi)
    abs_errors[i] = abs(uhs[i] - fxi)
  end

  # Plotting the absolute error
  plot(xs, abs_errors, seriestype = :scatter, label = "Absolute Errors", xlabel = "x", ylabel = "Absolute Error", size=(800, 800))
  savefig("Galerkin-generic-ode-absolute-errors-graph.png")
end


# Computes the error for a system of ne elements
function error_of_system(ne)

  # Solves the linear system
  K = init_K_matrix(ne)
  F = init_F_vector(ne)
  cs = K \ F

  # Computes the error
  error = calc_error(cs, ne)

  return error
end


# Plots the graph of errors according to the varying of n
function error_analysis()
  # Initializes the vectors
  hs = zeros(b - a + 1)
  hs2 = zeros(b - a + 1)
  errors = zeros(b - a + 1)

  # Computes the errors varying according to the variation of h
  for i in a:b
    # Initializes ne and h
    ne = (1 << i) - 1
    h = 1 / ne

    # Stores h and h^2 in lists
    hs[i-a+1] = h
    hs2[i-a+1] = h * h

    # Stores the error in a list
    error = error_of_system(ne)
    errors[i-a+1] = error
  end

  # Plots the errors in the graphic in a log scale
  plot(hs, [errors hs2], seriestype = [:scatter :line], label = ["Graph found by varying #h and computing the error" "h^2"], 
      xlabel = "h", ylabel = "error", size=(800, 800), xscale=:log10, yscale=:log10)

  # Saves the graph in a png file
  savefig("Galerkin-generic-ode-errors.png")
end


# Variables
alpha = 1                               # constant value
beta = 1                                # constant value
gamma = 1                               # constant value
u = (x) -> sin(pi * x)                  # function we are trying to approximate (we have it defined here just to compute the error)
f = (x) -> alpha*pi^2 * sin(pi*x) + beta*sin(pi*x) + gamma*pi*cos(pi*x)
x0 = 0                                  # starting point
a = 1                                   # lower limit to test the error varying
b = 14                                  # upper limit to test the error varying


# Plots the error analysis
error_analysis()

# Plots a comparison graph containing errors and an approximation comparison with ne as argument
plot_comparisson(5)