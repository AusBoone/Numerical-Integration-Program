# Numerical Integration

## Overview
This program numerically approximates the definite integral of a user-specified mathematical function over a given range. It supports three widely-used numerical integration methods: Composite Simpson’s Rule, Composite Trapezoidal Rule, and Composite Midpoint Rule. The program also offers graphical visualizations of the function and integration area, allowing users to better understand the numerical process and results.

## Features
- **Integration Methods**:
  - **Composite Simpson’s Rule**: A highly accurate method that requires an even number of subintervals.
  - **Composite Trapezoidal Rule**: A versatile method suitable for any number of subintervals.
  - **Composite Midpoint Rule**: Evaluates the function at the midpoint of each subinterval for a straightforward approximation.
- **User-Friendly Inputs**:
  - Accepts mathematical functions defined in terms of `x` (e.g., `x**2`, `sin(x)`, `exp(x)`).
  - Allows user-defined integration limits and the number of subintervals (`n`).
- **Error Estimation**:
  - Provides an estimate of the error by comparing results from different numerical methods.
- **Graphical Visualizations**:
  - Plots the function with the integration area shaded, offering a visual representation of the computation.
- **Secure Function Evaluation**:
  - Ensures safe execution of user-defined mathematical expressions using Python's Abstract Syntax Trees (AST).
- **Interactive Design**:
  - Step-by-step prompts guide the user through function definition, limit selection, method choice, and subinterval setup.
- **Customizable and Extensible**:
  - The modular code structure makes it easy to extend functionality or integrate with other projects.

## Prerequisites
- **Python Version**: Python 3.6 or later
- **Required Libraries**:
  - `numpy`: For numerical operations and calculations.
  - `matplotlib`: For plotting functions and visualizing the integration area.
