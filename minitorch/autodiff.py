from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    vals_plus = list(vals)
    vals_minus = list(vals)
    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon
    return (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative with respect to this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for this variable."""
        ...

    def is_leaf(self) -> bool:
        """Checks if this variable is a leaf in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Checks if this variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns an iterable of this variable's parents in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients for this variable's parents."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    visited = set()
    ret = []

    def visit(v: Variable) -> None:
        if v.unique_id not in visited:
            visited.add(v.unique_id)
            for p in v.parents:
                visit(p)
            ret.append(v)

    visit(variable)
    return reversed(ret)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Performs backpropagation through the computation graph.

    This function propagates the gradient backwards through the computation graph,
    starting from the given variable and using the provided derivative.

    Args:
    ----
        variable (Variable): The starting variable for backpropagation.
        deriv (Any): The initial derivative to propagate.

    Returns:
    -------
        None

    """
    # TODO: Implement for Task 1.4.
    order = list(topological_sort(variable))
    derivatives = {variable.unique_id: deriv}

    for var in order:
        current_deriv = derivatives.pop(var.unique_id, None)

        if current_deriv is not None:
            if var.is_leaf():
                var.accumulate_derivative(current_deriv)
            else:
                for parent, d_parent in var.chain_rule(current_deriv):
                    if parent.unique_id in derivatives:
                        derivatives[parent.unique_id] += d_parent
                    else:
                        derivatives[parent.unique_id] = d_parent


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Property that returns the saved values as a tuple of tensors.

        Returns
        -------
            Tuple[Any, ...]: A tuple containing the values saved during the forward pass.

        """
        return self.saved_values
