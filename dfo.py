# This is a complete implementation of a DFO algorithm in
# less than 100 lines of hopefully well-readable Python code.
#
# The MIT License (MIT)
#
# Copyright (C) 2025-2026 Tobias Glasmachers
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np

def fit_quadratic(Y, fY):                            # minimum norm fit of a quadratic model m(y) = g^T y + 1/2 y^T H y to data
	Y = Y[1:,]                                       # remove the first equation, which is trivial by design
	f = fY[1:]
	scale = np.max(np.linalg.norm(Y, axis=1))        # improve numerics through scaling
	Z = Y / scale
	k = Z.shape[0]; n = Z.shape[1]                   # Powell's method, but without keeping track of the inverse
	W = np.zeros((k+n, k+n))
	W[:k, :k] = (Z @ Z.T)**2
	W[:k, k:] = Z
	W[k:, :k] = Z.T
	c = np.concatenate((f, np.zeros(n)))
	coeff = np.linalg.solve(W, c)
	g = coeff[k:]
	H = np.sum([2*coeff[i] * np.outer(Z[i], Z[i]) for i in range(k)], axis=0)
	return g / scale, H / scale**2                   # undo scaling of the result

def min_2D(g2, H2, radius):                          # solve 2D trust region problem: min x@g2 + x@H2@x/2 s.t. norm(x) <= radius
	boundary = True                                  # flag: solution on the boundary
	if np.min(np.linalg.eigh(H2)[0]) > 0:            # strictly positive definite case
		v = -np.linalg.solve(H2, g2)                 # unconstrained candidate point
		boundary = (np.linalg.norm(v) >= radius)     # check whether it is feasible
		if boundary: v *= radius / np.linalg.norm(v) # project to the feasible region
	if boundary:
		def model(angle):                            # value of the quadratic model on the boundary
			x = [radius*np.cos(angle), radius*np.sin(angle)]
			return x @ g2 + 0.5 * (x @ H2 @ x)

		md = [model(np.pi*i/4) for i in range(8)]    # evaluate an "initial design" of 8 points on the quadratic model
		best = np.argmin(md)                         # index of the best initial point

		phi = (np.sqrt(5) - 1) / 2; psi = 1 - phi    # golden section search (GSS) for the optimal angle
		l = np.pi * (best-1)/4; ml = md[(best+7)%8]  # left boundary with model value
		r = np.pi * (best+1)/4; mr = md[(best+1)%8]  # right boundary with model value
		a = phi*l + psi*r; ma = model(a)             # "left interior" point of GSS
		b = psi*l + phi*r; mb = model(b)             # "right interior" point of GSS
		for _ in range(30):
			if ma < mb:
				r = b; mr = mb                       # drop the ...
				b = a; mb = ma                       # ... right point
				a = phi*l + psi*r; ma = model(a)     # evaluate a new point
			else:
				l = a; ml = ma                       # drop the ...
				a = b; ma = mb                       # ... left point
				b = psi*l + phi*r; mb = model(b)     # evaluate a new point
		angle = a if ma < mb else b
		v = radius * np.array([np.cos(angle), np.sin(angle)])
	return v, boundary

def sortInPlaceByFirst(arrays):                      # sort all arrays according to the first one in the list
	order = np.argsort(arrays[0])                    # sort the first array
	for a in arrays: a[:] = a[order]                 # apply the same order in-place to all arrays

def dfo(f, x, radius, precision = 1e-8):             # clean and simplified DFO algorithm
	n = len(x)                                       # problem dimension
	Y = np.concatenate(([x], x + radius * np.eye(n), x - radius * np.eye(n)), axis=0)   # initial set of points
	fY = np.array([f(y) for y in Y])                 # evaluate initial set
	evaluations = 2*n+1                              # number of function evaluations
	sortInPlaceByFirst((fY, Y))                      # sort by objective value
	c = np.copy(Y[0]); fc = fY[0]                    # incumbent with objective value
	Y -= c; fY -= fc                                 # shift incumbent to the origin and to objective value 0
	g, H = fit_quadratic(Y, fY)                      # fit the initial model

	while g@g > precision**2 and radius > precision: # main loop: stop when the model becomes flat or the radius becomes small
		# projection to 2D subspace spanned by model gradient and unconstrained optimum with positive definite Hessian
		minEV = np.min(np.linalg.eigh(H)[0])         # minimal eigenvalue of H
		P = np.stack((g, np.linalg.solve(H + 1.5 * max(0, -minEV) * np.eye(n), g)))
		P[0] /= np.linalg.norm(P[0])                 # Gram-Schmidt orthogonalization
		P[1] -= (P[1]@P[0]) * P[0]
		P[1] /= np.linalg.norm(P[1])
		g2 = P @ g; H2 = P @ H @ P.T                 # project the model: g2 is a 2D vector, H2 a 2x2 matrix

		v, boundary = min_2D(g2, H2, radius)         # trust region step on the 2D subspace
		y0 = P.T @ v                                 # candidate step in the original space
		fy0 = f(c + y0) - fc; evaluations += 1       # evaluate the objective function
		my0 = g @ y0 + 0.5 * (y0 @ H @ y0)           # compute the model prediction

		rho = (0 - fy0) / (0 - my0)                  # model reliability
		if my0 >= 0 or rho < 0.1: radius /= 2        # adapt the trust region radius
		elif rho > 0.75 and boundary: radius *= 2

		Y[-1] = y0; fY[-1] = fy0                     # overwrite the worst point
		res = np.zeros(2*n+1)                        # compute residuals (objective minus model)
		res[-1] = fy0 - my0

		if my0 >= 0 or rho < 0.25:                   # model improvement: overwrite second-worst point
			Y[-2] = radius * np.random.randn(n)      # multi-variate Gaussian random sample
		fY[-2] = f(c + Y[-2]) - fc; evaluations += 1
		res[-2] = fY[-2] - (g @ Y[-2] + 0.5 * (Y[-2] @ H @ Y[-2]))

		sortInPlaceByFirst((fY, Y, res))             # sort by objective value
		if fY[0] < 0:                                # improvement found
			dy = np.copy(Y[0])
			c += dy                                  # shift the incumbent
			fc += fY[0]
			Y -= dy                                  # shift the set of points
			fY -= fY[0]
			res -= res[0]                            # shift the residuals
			g += H @ dy                              # update the model accordingly
		dg, dH = fit_quadratic(Y, res)               # minimum norm model update based on residuals
		g += dg; H += dH
	return c, fc, evaluations                        # return incumbent, its objective value, and the number of function evaluations




#######################################################################
# apply the solver to a randomly rotated Rosenbrock problem
#

U = None
def rotate(x):
	global U
	if U is None:
		n = len(x)
		H = np.random.randn(n, n)
		u, s, vh = np.linalg.svd(H, full_matrices=False)
		U = u @ vh
	return U @ x

def rosenbrock(x):
	n = len(x)
	return np.sum(100.0 * np.square(x[1:n] - np.square(x[0:n-1])) + np.square(x[0:n-1] - np.ones(n-1)))

x0 = np.random.randn(20)
objective = rosenbrock

print("DFO solver on", objective.__name__, "in dimension", len(x0))
x, fx, evaluations = dfo(objective, x0, 0.1)
print("# function evaluations =", evaluations)
print("   x =", np.round(x, 7))
print("f(x) =", np.round(fx, 12))
