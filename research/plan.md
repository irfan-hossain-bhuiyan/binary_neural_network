# Research roadmap

This document separates recorded evidence from deductions, working hypotheses, and
possible future work. Numerical claims live in the canonical result JSON files.

## Current objectives

The project now treats these as four different objectives:

1. **Function learning:** the continuous model computes the target truth table.
2. **Continuous/output polarization:** predictions move toward 0 or 1.
3. **Parameter polarization:** gates and biases move away from their thresholds.
4. **Correct discrete topology:** thresholding parameters and running the exact
   Boolean model still computes the target function.

The experiments show that these objectives are not equivalent. Low continuous
loss does not guarantee Boolean recovery. Endpoint-consistent operators do not
make every finite interior solution threshold-compatible. Near-binary parameters
can encode the wrong topology. Near-binary outputs can also encode the wrong
topology. Continuous/discrete agreement can be excellent for the wrong function;
A5 is the clearest example.

## Mathematical foundations

### Continuous XOR

\[
x\oplus_c b=x+b-2xb.
\]

Away from exact threshold ties, thresholding commutes with this XOR:
\(T(x\oplus_c b)=T(x)\oplus T(b)\). This makes XOR unusually well behaved.

### Product AND mismatch

For \(v=ag\), thresholding does not generally commute with AND. For example,
\(a=g=.7\) gives \(ag=.49\), while both operands threshold to one.

### Lehmer-p2 OR

\[
L_2(v)=\frac{\sum_i v_i^3}{\sum_i v_i^2}.
\]

It is endpoint-rigid but does not commute with thresholded Boolean OR. The only
possible threshold disagreement is \(L_2(v)<.5\) with \(\max_i v_i\ge .5\),
which was observed in I6.

### XOR residual gradient

For \(y=x+F(x)-2xF(x)\),
\[
J_y=\operatorname{diag}(1-2F)+\operatorname{diag}(1-2x)J_F.
\]
The direct path is \(1-2F\): it is approximately +1 near \(F=0\), -1 near
\(F=1\), and suppressed near \(F=.5\). A2 supports the hypothesis that the
path becomes stronger after branch polarization; this is not a universal proof
that XOR residuals outperform all alternatives.

## Loss evidence

- **MSE** is target-directed and reliably trains the deterministic tasks better
  than L1 in this project, but its conditional optimum is \(p^*=q\) under label
  uncertainty.
- **BCE** also has \(p^*=q\); its difference is gradient geometry, not a
  Boolean statistical optimum.
- **L1** is convex. At \(q=.5\), every \(p\in[0,1]\) minimizes its population
  risk, and its per-example gradient has constant magnitude away from zero.
  Poor behavior cannot be explained by non-convexity.
- **Power losses** \(|p-y|^\alpha\), \(1<\alpha<2\), sharpen the population
  optimum toward a majority endpoint. They helped selected XOR basins but did
  not transfer cleanly to one-hot MNIST.
- **Output Gini** \(p(1-p)\), tested in A5 with \(\lambda=1.5\), strongly
  polarizes outputs. Fixed Gini sent about 97.6% of outputs to below .01 or
  above .99 without producing an exact adder. Output polarization is therefore
  mechanical polarization, not topology credit. A5 also found late conflict
  between MSE and Gini gradients.

## Error-distribution polarization hypothesis

For per-row MSE \(e_n=\frac15\sum_j(p_{nj}-y_{nj})^2\), consider
\[
\phi_\lambda(e)=e+\lambda e(1-e).
\]
Then \(\phi'(e)=1+\lambda-2\lambda e\), \(\phi''(e)=-2\lambda\). For
\(0<\lambda<1\), every error reduction improves the objective, while the
objective is concave in row error. Since
\[
E[\phi(e)]=\mu+\lambda\mu(1-\mu)-\lambda\operatorname{Var}(e),
\]
it prefers larger error variance at matched mean. The desired interpretation
is many nearly-perfect rows plus a minority of difficult rows. This remains a
hypothesis until A6.

## Evidence that weakened prior approaches

These are scoped experimental results, not universal impossibility proofs.

| approach | scope of negative result |
|---|---|
| old softmax OR semantic gap | failed to provide a reliable exact Boolean path in the tested XOR/addition studies |
| sparse mean-field initialization | worsened optimization in the tested initialization comparisons |
| stronger gate polarization | hardened wrong topology in I5 |
| Lehmer→max homotopy | did not repair the known XOR topology in I7 |
| global Boolean STE scaling | too weak for the tested I9 seed3 continuation |
| static layer-balanced STE | too aggressive/unstable in the tested I10 setting |
| fixed power losses | seed/basin sensitive; XOR gains did not transfer cleanly to MNIST |
| alpha curriculum | did not improve XOR topology selection in I13 |
| one-hot MNIST POWER | unbalanced POWER collapsed toward zero-hot outputs in M1 |
| row-max tolerance | changed whole-word error behavior without solving topology in A3/A4 |
| output Gini | polarized outputs but hardened incorrect addition topology in A5 |

## Possible future approaches

These are backlog ideas, not commitments.

- **Error-distribution polarization:** immediate A6 test.
- **Gradient-safe polarization:** project or limit a conflicting polarization
  gradient. A possible bound is
  \(\lambda_{max}=\beta\|g_T\|^2/(-g_T^\top g_R)\) when the dot product is
  negative. Do not implement before evidence supports it.
- **Proximal polarization/quantization:** perform a task step, then a separate
  move toward the discrete set; ProxQuant is relevant prior work.
- **Exact Boolean forward with surrogate backward:** I8 gave directional XOR
  evidence; future scaling must avoid I10's static layer-scaling failure.
- **Stochastic topology:** sampled binary/concrete or hard-concrete gates could
  give both sides of a decision signal, at the cost of variance and machinery.
- **Alternative discrete parameterization:** compare explicit Boolean-gate
  distributions, conceptually related to DDLGN, only after this architecture
  is diagnosed.
- **Threshold-homomorphic operators:** seek operators for which
  \(T(F_c(x))\approx F_b(T(x))\) through useful interiors. XOR is unusually
  close; product AND and Lehmer OR are not.
- **Exact topology neighborhoods:** count useful one-bit flips around a trained
  circuit and distinguish missing credit from a coordinated local minimum.
- **Scaling benchmark:** after reliable 4-bit addition, test exhaustive
  6-bit + 6-bit → 7-bit (4096 rows), not an automatic next run.
- **Classification later:** revisit MNIST with grouped Boolean vote neurons and
  measure vote polarization and continuous/discrete class-score gaps.

## Representational-capacity gate

Before treating further 4-bit failures as optimization failures, prove that the
implemented exact discrete architecture can represent 4-bit addition:
8 → 64 stem, two width-64 XOR-residual blocks, and a 64 → 5 head. A valid
capacity result must execute the actual `DiscreteModernLogicGateNet` on all 256
rows with bit and exact-row accuracy 1.0. Generic universal-approximation
arguments are insufficient. If this gate fails, stop optimization experiments
and redesign the architecture.

### Capacity gate result (2026-09-25)

This gate is passed. `research/capacity_a4.py` constructs an explicit
parameterization of the actual model using stem constants/literals, NAND
monomials, ORs of complemented monomials for ripple carries, and XOR residual
wiring for the sum bits. The integer and ripple-carry target generators agree,
and the construction obtains 1.0 Boolean bit accuracy and 1.0 exact-row
accuracy on all 256 inputs. This establishes existence inside the implemented
class, not trainability; the compact construction and reproduction command
are documented in `research/capacity_a4_report.md`.

## Methodology and decision gates

Every comparison measures solution quality and training speed, never raw native
loss across different families. Track continuous and Boolean exactness,
stability, continuous→Boolean disagreement, milestone times, carry-chain
behavior, gradient propagation, and topology movement. Preserve negative
results and avoid heavyweight checkpoints.

1. If the current architecture cannot represent exact 4-bit addition, stop and
   redesign it.
2. If A6 changes error distributions but not Boolean topology, stop inventing
   output losses and move to topology credit.
3. If topology-aware methods work on 4-bit addition, test 6-bit addition.
4. A one-seed result is screening evidence, not a success claim.
5. Do not return to MNIST until deterministic compositional tasks have a
   reproducible continuous→Boolean story.
