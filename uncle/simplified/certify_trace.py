"""Diagnostics for the ``certify()`` predicate in ``model_certify``."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from .model_certify import _validate_gamma, certify


def build_certify_trace(
    *,
    G_op: float,
    G_D: float,
    G_O: float,
    A_2: float,
    A_inf: float,
    B_2: float,
    B_inf: float,
    R_2: float,
    R_inf: float,
    Lambda_off: float,
    gamma: float = 1.0,
) -> dict[str, Any]:
    """Evaluate ``certify()`` inputs and per-condition margins.

    Inputs are raw constants plus the curvature-Lipschitz constant ``gamma``;
    each condition's lhs below uses the γ-scaled drift products,
    matching ``certify(gamma=...)``. Condition NAME strings are stable
    identifiers (plot/report code keys on them) — they keep the γ=1 spelling
    even though the lhs values include γ.
    """
    gamma = _validate_gamma(gamma)
    inputs = {
        "G_op": float(G_op),
        "G_D": float(G_D),
        "G_O": float(G_O),
        "A_2": float(A_2),
        "A_inf": float(A_inf),
        "B_2": float(B_2),
        "B_inf": float(B_inf),
        "R_2": float(R_2),
        "R_inf": float(R_inf),
        "Lambda_off": float(Lambda_off),
        "gamma": float(gamma),
    }

    S_2 = A_2 + R_2
    S_inf = A_inf + R_inf
    derived = {"S_2": float(S_2), "S_inf": float(S_inf)}

    # Scaled drift constants used by every condition below (single-scaling:
    # raw values live in ``inputs``; conditions consume the γ-products).
    gG_op = gamma * G_op
    gG_D = gamma * G_D
    gLambda_off = gamma * Lambda_off

    conditions: list[dict[str, Any]] = []

    def _add(name: str, lhs: float, rhs: float, strict: bool = True) -> bool:
        margin = rhs - lhs
        passed = lhs < rhs if strict else lhs <= rhs
        conditions.append(
            {
                "name": name,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "margin": float(margin),
                "passed": bool(passed),
            }
        )
        return passed

    guard_failed = R_2 <= 0 or R_inf <= 0 or not (
        np.isfinite(R_2) and np.isfinite(R_inf)
    )
    if guard_failed:
        accepted = False
    else:
        c1 = _add("G_op * S_inf < 1", gG_op * S_inf, 1.0)
        c2 = _add("G_D * S_inf < 1", gG_D * S_inf, 1.0)
        denom_op = 1.0 - gG_op * S_inf
        denom_d = 1.0 - gG_D * S_inf
        if c1 and denom_op > 0:
            c3_lhs = B_2 / denom_op
            c3 = _add("B_2 / (1 - G_op*S_inf) < R_2", c3_lhs, R_2)
        else:
            c3 = False
            conditions.append(
                {
                    "name": "B_2 / (1 - G_op*S_inf) < R_2",
                    "lhs": float("nan"),
                    "rhs": float(R_2),
                    "margin": float("nan"),
                    "passed": False,
                }
            )
        if c1 and c2 and denom_op > 0 and denom_d > 0:
            c4_lhs = (B_inf + gLambda_off * B_2 / denom_op) / denom_d
            c4 = _add(
                "(B_inf + Lambda_off*B_2/(1-G_op*S_inf)) / (1-G_D*S_inf) < R_inf",
                c4_lhs,
                R_inf,
            )
        else:
            c4 = False
            conditions.append(
                {
                    "name": "(B_inf + Lambda_off*B_2/(1-G_op*S_inf)) / (1-G_D*S_inf) < R_inf",
                    "lhs": float("nan"),
                    "rhs": float(R_inf),
                    "margin": float("nan"),
                    "passed": False,
                }
            )
        accepted = c1 and c2 and c3 and c4

    certify_result = certify(
        G_op=G_op,
        G_D=G_D,
        A_inf=A_inf,
        B_2=B_2,
        B_inf=B_inf,
        R_2=R_2,
        R_inf=R_inf,
        Lambda_off=Lambda_off,
        gamma=gamma,
    )

    return {
        "inputs": inputs,
        "derived": derived,
        "conditions": conditions,
        "guard_failed": guard_failed,
        "accepted": accepted,
        "certify_matches": accepted == certify_result,
    }


def print_certify_trace(
    trace: Mapping[str, Any],
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> None:
    """Print a human-readable certify diagnostic block to stdout."""
    ctx = dict(context or {})
    header = ctx.get("label", "CERTIFY")
    print(f"\n{'=' * 70}")
    print(f"  {header}")
    for key in ("n", "d", "n_ret", "k", "lam", "safety_factor", "R_2"):
        if key in ctx:
            val = ctx[key]
            if key == "lam":
                print(f"  {key} = {val:.6e}")
            elif key in ("n", "d", "n_ret", "k"):
                print(f"  {key} = {val}")
            else:
                print(f"  {key} = {val:.6g}")
    print(f"{'=' * 70}")

    inputs = trace["inputs"]
    derived = trace["derived"]
    print(
        f"  R_2 = {inputs['R_2']:.6e}   "
        f"R_inf = {inputs['R_inf']:.6e}   "
        f"Lambda_off = {inputs['Lambda_off']:.6e}"
    )
    if inputs.get("gamma", 1.0) != 1.0:
        print(f"  gamma (curvature-Lipschitz) = {inputs['gamma']:.6e} — "
              "condition lhs values below use gamma-scaled drift products")
    print("\n  Inputs to certify():")
    for name in (
        "G_op",
        "G_D",
        "G_O",
        "A_2",
        "A_inf",
        "B_2",
        "B_inf",
    ):
        print(f"    {name:6s} = {inputs[name]:.6e}")

    print("\n  Derived:")
    print(f"    S_2   = A_2 + R_2 = {derived['S_2']:.6e}")
    print(f"    S_inf = A_inf + R_inf = {derived['S_inf']:.6e}")

    print("\n  Conditions:")
    for cond in trace["conditions"]:
        status = "PASS" if cond["passed"] else "FAIL"
        print(f"    {cond['name']}")
        print(
            f"      lhs = {cond['lhs']:.6e}   rhs = {cond['rhs']:.6e}   "
            f"margin = {cond['margin']:+.6e}   {status}"
        )

    if trace.get("guard_failed"):
        print("\n  Guard: R_2/R_inf invalid — rejected before conditions")

    result = "PASS" if trace["accepted"] else "FAIL"
    print(f"\n  RESULT: {result}")
    if not trace.get("certify_matches", True):
        print("  WARNING: trace accepted flag disagrees with certify()")
    print(f"{'=' * 70}\n")
