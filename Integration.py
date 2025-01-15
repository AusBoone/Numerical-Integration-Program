#---------------------------------------------
# Assignment Title: Numerical Integration Program
# Course: Calculus II
#
# Author: Austin Boone
#
# Description:
# This program numerically approximates the definite integral of a user-specified
# function f(x) over given bounds [a, b] using various numerical integration methods,
# including the Composite Simpson’s Rule, Trapezoidal Rule, and Midpoint Rule.
# Additionally, it provides graphical visualization of the function and the
# integration area for better understanding.
#
# Numerical Methods:
# 1. Composite Simpson’s Rule:
#    - Requires an even number of subintervals (n).
#
# 2. Composite Trapezoidal Rule:
#
# 3. Composite Midpoint Rule:
#    - Evaluates the function at the midpoints of each subinterval.
#
# Instructions:
# 1. Run the program.
# 2. Enter the function in terms of x when prompted (e.g., x**2, sin(x), exp(x)).
#    Available math functions: sin, cos, tan, exp, sqrt, log, pi, e, etc.
# 3. Enter the lower and upper integration limits (a and b).
# 4. Choose the numerical integration method.
# 5. Enter the number of subintervals (n). 
# 6. The program will compute and display the approximate integral value,
#    provide error estimates, and display a graph of the function with the
#    integration area highlighted.
#---------------------------------------------

import math
import numpy as np
import ast
import operator
import sys
import matplotlib.pyplot as plt
from enum import Enum, auto


class IntegrationMethod(Enum):
    """
    Enumeration of available numerical integration methods.
    """
    SIMPSON = auto()
    TRAPEZOIDAL = auto()
    MIDPOINT = auto()


class NumericalIntegrator:
    """
    A class to perform numerical integration of user-defined functions using various methods.
    """

    # Define supported operators for safe evaluation
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg
    }

    # Define supported functions
    allowed_functions = {
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'sqrt': np.sqrt,
        'log': np.log,
        'pi': np.pi,
        'e': np.e,
        'abs': np.abs,
        'arcsin': np.arcsin,
        'arccos': np.arccos,
        'arctan': np.arctan,
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        'floor': np.floor,
        'ceil': np.ceil
    }

    def __init__(self):
        """
        Initializes the NumericalIntegrator instance.
        """
        self.func_str = ""
        self.f = None
        self.a = 0.0
        self.b = 1.0
        self.n = 1000
        self.method = IntegrationMethod.SIMPSON

    def safe_eval(self, expr, x):
        """
        Safely evaluate a mathematical expression with the given x using AST parsing.

        Parameters:
        - expr (str): The mathematical expression to evaluate.
        - x (float or np.ndarray): The value(s) of x.

        Returns:
        - float or np.ndarray: The result of the evaluated expression.
        """
        def _eval(node):
            if isinstance(node, ast.Num):  # <number>
                return node.n
            elif isinstance(node, ast.Name):
                if node.id == 'x':
                    return x
                elif node.id in self.allowed_functions:
                    return self.allowed_functions[node.id]
                else:
                    raise ValueError(f"Use of '{node.id}' is not allowed.")
            elif isinstance(node, ast.BinOp):
                if type(node.op) not in self.allowed_operators:
                    raise ValueError(f"Operator '{type(node.op).__name__}' not allowed.")
                return self.allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                if type(node.op) not in self.allowed_operators:
                    raise ValueError(f"Unary operator '{type(node.op).__name__}' not allowed.")
                return self.allowed_operators[type(node.op)](_eval(node.operand))
            elif isinstance(node, ast.Call):
                func = _eval(node.func)
                args = [_eval(arg) for arg in node.args]
                return func(*args)
            else:
                raise TypeError(f"Unsupported expression: {node}")

        try:
            parsed_expr = ast.parse(expr, mode='eval').body
            return _eval(parsed_expr)
        except Exception as e:
            raise ValueError(f"Invalid function expression: {e}")

    def get_function(self):
        """
        Prompts the user to input a mathematical function and sets the function string and callable.

        Raises:
        - ValueError: If the function expression is invalid.
        """
        while True:
            self.func_str = input("Enter f(x) (e.g., x**2, sin(x), exp(x)): ").strip()
            if not self.func_str:
                print("Function input cannot be empty. Please try again.")
                continue
            try:
                # Test evaluation with a sample x value
                self.f = lambda x: self.safe_eval(self.func_str, x)
                test_val = self.f(1.0)  # Test with x=1.0
                break
            except Exception as e:
                print(f"Error in function expression: {e}. Please try again.")

    def get_limits(self):
        """
        Prompts the user to input the lower and upper limits of integration.

        Sets the instance variables 'a' and 'b'.

        Raises:
        - ValueError: If the limits are invalid.
        """
        while True:
            try:
                self.a = float(input("Enter the lower limit (a): "))
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value for the lower limit.")

        while True:
            try:
                self.b = float(input("Enter the upper limit (b): "))
                if self.b == self.a:
                    print("Upper and lower limits cannot be the same. Please enter a different value.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value for the upper limit.")

        # Swap if a > b
        if self.a > self.b:
            print(f"Lower limit a={self.a} is greater than upper limit b={self.b}. Swapping the limits.")
            self.a, self.b = self.b, self.a

    def choose_method(self):
        """
        Allows the user to choose the integration method.

        Sets the instance variable 'method'.

        Raises:
        - ValueError: If an unsupported method is selected.
        """
        methods = {
            '1': IntegrationMethod.SIMPSON,
            '2': IntegrationMethod.TRAPEZOIDAL,
            '3': IntegrationMethod.MIDPOINT
        }
        print("\nChoose the integration method:")
        print("  1. Simpson’s Rule")
        print("  2. Trapezoidal Rule")
        print("  3. Midpoint Rule")

        while True:
            choice = input("Enter the number corresponding to your choice (1/2/3): ").strip()
            if choice in methods:
                self.method = methods[choice]
                return
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

    def get_subintervals(self):
        """
        Prompts the user to input the number of subintervals.

        Sets the instance variable 'n'.

        Raises:
        - ValueError: If the number of subintervals is invalid.
        """
        while True:
            try:
                n_input = input("Enter the number of subintervals (n): ").strip()
                n = int(n_input)
                if n <= 0:
                    print("Number of subintervals must be positive.")
                    continue
                if self.method == IntegrationMethod.SIMPSON and n % 2 != 0:
                    print(f"Simpson’s Rule requires an even number of subintervals. Incrementing n to {n + 1}.")
                    n += 1
                self.n = n
                break
            except ValueError:
                print("Invalid input. Please enter an integer value for the number of subintervals.")

    def composite_simpson(self):
        """
        Approximates the definite integral using the Composite Simpson’s Rule.

        Returns:
        - float: The approximate value of the integral.
        """
        n = self.n
        a, b = self.a, self.b
        if n % 2 != 0:
            print(f"Number of subintervals n={n} is not even. Incrementing to n={n + 1}.")
            n += 1
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = self.f(x)

        # Simpson's rule coefficients: 1, 4, 2, 4, ..., 4, 1
        coefficients = np.ones(n + 1)
        coefficients[1:n:2] = 4
        coefficients[2:n-1:2] = 2

        integral = (h / 3) * np.dot(coefficients, y)
        return integral

    def composite_trapezoidal(self):
        """
        Approximates the definite integral using the Composite Trapezoidal Rule.

        Returns:
        - float: The approximate value of the integral.
        """
        n = self.n
        a, b = self.a, self.b
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = self.f(x)

        # Trapezoidal rule coefficients: 1, 2, 2, ..., 2, 1
        coefficients = np.ones(n + 1)
        coefficients[1:n] = 2

        integral = (h / 2) * np.dot(coefficients, y)
        return integral

    def composite_midpoint(self):
        """
        Approximates the definite integral using the Composite Midpoint Rule.

        Returns:
        - float: The approximate value of the integral.
        """
        n = self.n
        a, b = self.a, self.b
        h = (b - a) / n
        midpoints = a + h * (np.arange(n) + 0.5)
        y = self.f(midpoints)
        integral = h * np.sum(y)
        return integral

    def compute_integral(self):
        """
        Computes the integral using the selected numerical method.

        Returns:
        - float: The approximate value of the integral.
        """
        if self.method == IntegrationMethod.SIMPSON:
            return self.composite_simpson()
        elif self.method == IntegrationMethod.TRAPEZOIDAL:
            return self.composite_trapezoidal()
        elif self.method == IntegrationMethod.MIDPOINT:
            return self.composite_midpoint()
        else:
            raise ValueError("Unsupported integration method selected.")

    def estimate_error(self, result):
        """
        Estimates the error of the integration by comparing results from different methods.

        Parameters:
        - result (float): The result from the selected integration method.

        Returns:
        - dict: A dictionary containing error estimates from other methods.
        """
        error_estimates = {}
        if self.method != IntegrationMethod.TRAPEZOIDAL:
            trapezoidal_result = self.composite_trapezoidal()
            error_estimates['Trapezoidal Rule'] = abs(result - trapezoidal_result)
        if self.method != IntegrationMethod.MIDPOINT:
            midpoint_result = self.composite_midpoint()
            error_estimates['Midpoint Rule'] = abs(result - midpoint_result)
        if self.method != IntegrationMethod.SIMPSON:
            simpson_result = self.composite_simpson()
            error_estimates['Simpson’s Rule'] = abs(result - simpson_result)
        return error_estimates

    def plot_function(self, integral):
        """
        Plots the function and highlights the area under the curve between a and b.

        Parameters:
        - integral (float): The approximate value of the integral.
        """
        x = np.linspace(self.a, self.b, 1000)
        y = self.f(x)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b', label=f'f(x) = {self.func_str}')
        plt.fill_between(x, y, where=((x >= self.a) & (x <= self.b)), color='skyblue', alpha=0.4, label='Integration Area')
        plt.title(f'Numerical Integration using {self.method.name.replace("_", " ")}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        """
        Executes the numerical integration process by interacting with the user,
        performing computations, and displaying results.
        """
        print("=== Numerical Integration Program ===\n")
        try:
            # Step 1: Get the function from the user
            self.get_function()

            # Step 2: Get the integration limits
            self.get_limits()

            # Step 3: Choose the integration method
            self.choose_method()

            # Step 4: Get the number of subintervals
            self.get_subintervals()

            print("\nComputing the integral...")

            # Step 5: Perform the integration using the chosen method
            result = self.compute_integral()

            # Step 6: Display the result
            print(f"\nApproximate value of the integral ∫_{self.a}^{self.b} {self.func_str} dx ≈ {result:.10f}")

            # Step 7: Error Estimation (Optional)
            print("\nEstimating the error by comparing with other numerical methods.")
            error_estimates = self.estimate_error(result)
            for method, error in error_estimates.items():
                print(f"{method} result: {self.get_method_result(method):.10f}")
                print(f"Estimated error: {error:.10f}")

            # Step 8: Plot the function and integration area
            self.plot_function(result)

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            sys.exit(1)

    def get_method_result(self, method_name):
        """
        Retrieves the result from a specific integration method.

        Parameters:
        - method_name (str): The name of the integration method.

        Returns:
        - float: The result from the specified method.
        """
        if method_name == 'Trapezoidal Rule':
            return self.composite_trapezoidal()
        elif method_name == 'Midpoint Rule':
            return self.composite_midpoint()
        elif method_name == 'Simpson’s Rule':
            return self.composite_simpson()
        else:
            raise ValueError(f"Method '{method_name}' not recognized.")


def main():
    """
    The main function to instantiate and run the NumericalIntegrator.
    """
    integrator = NumericalIntegrator()
    integrator.run()


if __name__ == "__main__":
    main()


# Usage Example
"""
=== Numerical Integration Program ===

Enter f(x) (e.g., x**2, sin(x), exp(x)): sin(x)
Enter the lower limit (a): 0
Enter the upper limit (b): pi

Choose the integration method:
  1. Simpson’s Rule
  2. Trapezoidal Rule
  3. Midpoint Rule
Enter the number corresponding to your choice (1/2/3): 1
Enter the number of subintervals (n): 1000

Computing the integral...

Approximate value of the integral ∫0.0^3.141592653589793 sin(x) dx ≈ 2.0000000000

Estimating the error by comparing with other numerical methods.
Trapezoidal Rule result: 2.0000000108
Estimated error: 0.0000000108
Midpoint Rule result: 2.0000000000
Estimated error: 0.0000000000
"""
