from sympy import symbols, diff

x = symbols('x')
# Define the function
f = x**2 / (1 - x)

# Compute the first derivative
f_prime = diff(f, x)

# Compute the second derivative
f_double_prime = diff(f_prime, x)
f_double_prime.simplify()

print(f_double_prime.simplify())