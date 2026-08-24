# Exp - 6(AAI) - Solving a Stochastic Grid-World Markov Decision Process Using Value Iteration and Policy Iteration
## By Dr. N SARAVANAN, TSML006,ASSISTANT PROFESSOR,AIML,SEC
A compact Python implementation of two dynamic-programming methods for solving a stochastic grid-world Markov decision process (MDP):

- **Value Iteration**
- **Policy Iteration**

The program calculates the utility of each state and prints the resulting greedy policy.

## Why sequential decision problems matter

A **sequential decision problem** is one in which a decision made now affects what choices, rewards, and risks are available later. The objective is not simply to choose the action with the best immediate result; it is to choose actions that lead to the best *long-term* outcome despite uncertainty.

This grid world is a small example: moving toward the goal may be worthwhile, but a movement can drift unexpectedly toward the trap. The agent must therefore balance the immediate step cost, the chance of reaching the reward, and the risk of a future penalty.

Solving problems like this is important because many real systems operate through a sequence of connected decisions:

- **Robotics and navigation:** choosing safe routes while accounting for imperfect movement and obstacles.
- **Operations and logistics:** planning inventory, deliveries, and resources when demand and travel conditions are uncertain.
- **Finance:** weighing current returns against longer-term risk and future market outcomes.
- **Healthcare:** selecting treatments over time as a patient's condition and response evolve.
- **Reinforcement learning:** training an agent to act effectively from feedback rather than fixed instructions.

Value iteration and policy iteration provide principled ways to solve a Markov decision process (MDP). They evaluate both immediate rewards and expected future utility, producing a policy that tells an agent what action to take in each state. This makes them foundational techniques for planning under uncertainty.

## Mathematical formulation


| Symbol | Meaning |
| --- | --- |
| s| Current state |
| a | Action selected in state \(s\) |
| s'| Possible successor (next) state |
| R(s) | Immediate reward in state \(s\) |
| P(s' \| s, a) | Probability of reaching \(s'\) after taking \(a\) in \(s\). The vertical bar \| means “given.” |
| γ | Discount factor, which controls how much future rewards matter relative to immediate rewards. In this program, γ = 1.0. |

### Bellman optimality equation

The optimal utility of a state is its immediate reward plus the discounted expected utility of the best available action:

$$
V^{\star}(s) = R(s) + \gamma \max_{a \in A(s)} \sum_{s'} P(s' \mid s, a)V^{*}(s')
$$

### Value iteration update

Value iteration repeatedly applies the Bellman optimality update until successive utility values change by less than the chosen tolerance:

$$
V_{k+1}(s) = R(s) + \gamma \max_{a \in A(s)} \sum_{s'} P(s' \mid s, a)V_k(s')
$$

For this program, an intended action succeeds with probability \(0.8\), while the agent drifts left or right with probability \(0.1\) each. Therefore, the expected successor utility is:

$$
\sum_{s'} P(s' \mid s, a)V(s') =
0.8V(s_{\text{intended}}) +
0.1V(s_{\text{left}}) +
0.1V(s_{\text{right}})
$$

### Arbitrary initial policy

Policy iteration begins with any valid action at each non-terminal state:

$$
\pi_0(s) \in A(s), \quad \forall s \notin \text{Terminal States}
$$

This implementation initializes the policy with `UP` for every non-terminal state.

### Policy evaluation

For a fixed policy \(\pi\), the utility of each state is calculated as:

$$
V^{\pi}(s) = R(s) + \gamma \sum_{s'} P(s' \mid s, \pi(s))V^{\pi}(s')
$$

### Policy improvement

The policy is improved by selecting the action with the largest expected successor utility:

$$
\pi_{\text{new}}(s) =
\arg\max_{a \in A(s)}
\sum_{s'} P(s' \mid s, a)V^{\pi}(s')
$$

Policy evaluation and improvement repeat until the policy does not change:

$$
\pi_{\text{new}} = \pi
$$

## Grid-world configuration

The environment is a 3 × 4 grid:

```text
+-------+-------+-------+--------+
| (2,0) | (2,1) | (2,2) | Goal   |
+-------+-------+-------+--------+
| (1,0) | Block | (1,2) | Trap   |
+-------+-------+-------+--------+
| (0,0) | (0,1) | (0,2) | (0,3)  |
+-------+-------+-------+--------+
```

| Element | Details |
| --- | --- |
| Normal-state reward | `-0.04` |
| Goal at `(2, 3)` | terminal state with reward `+1.0` |
| Trap at `(1, 3)` | terminal state with reward `-1.0` |
| Blocked cell | `(1, 1)` |
| Discount factor | `γ = 1.0` |
| Convergence threshold | `ε = 1e-4` |

Coordinates use `(row, column)` with `(0, 0)` at the bottom-left. The displayed arrays are flipped vertically so the top row appears first.

## Movement model

The available actions are `UP`, `DOWN`, `LEFT`, and `RIGHT`.

An intended action succeeds with probability **0.8**. With probability **0.1** each, the agent drifts to the action on its left or right. If a move would leave the grid or enter the blocked cell, the agent stays in its current state.

## Requirements

- Python 3.8 or newer
- NumPy

Install the dependency:

```bash
python3 -m pip install numpy
```

## Run

```bash
python3 valuePolicyIter.py
```

By default, the script runs value iteration and prints a utility table followed by the extracted policy.

Example output:

```text
--- Final Utility Table ---
[[ 0.812  0.868  0.918  1.   ]
 [ 0.762  0.796  0.66  -1.   ]
 [ 0.705  0.655  0.611  0.388]]

--- Extracted Policy Layout ---
[['RIGHT' 'RIGHT' 'RIGHT' 'GOAL']
 ['UP' 'UP' 'UP' 'TRAP']
 ['UP' 'LEFT' 'LEFT' 'LEFT']]
```

### Value iteration (default)

```python
U, policy = value_iteration(gamma, epsilon)
```

Value iteration repeatedly applies the Bellman optimality update until the largest utility change is smaller than `epsilon`.

### Policy iteration

Comment out the value-iteration line and enable this one:

```python
U, policy = policy_iteration(gamma, epsilon)
```

Policy iteration alternates between evaluating the current policy and improving it until no action changes.

## Customize the environment

Edit these values near the top of the script to experiment with other MDPs:

- `rows`, `cols` — grid size
- `R` — reward table
- `terminals` — terminal-state coordinates
- `gamma` — discount factor
- `epsilon` — stopping tolerance
- `get_next_state()` — boundaries and blocked-cell behavior
- `get_action_distribution()` / `expected_utility()` — transition dynamics

## Notes

- Terminal-state utilities stay fixed at their assigned rewards.
- The policy grid includes a label for every non-terminal coordinate, including the blocked coordinate. Since that cell cannot be entered, its displayed action is not part of the reachable environment.

## PROGRAM
```python
import numpy as np

# -------------------------------------------------
# Grid World Configuration
# -------------------------------------------------

ROWS = 3
COLS = 4

gamma = 0.9              # Discount Factor
living_reward = -1       # Reward for every move
threshold = 0.001        # Convergence threshold


# -------------------------------------------------
# Special States
# -------------------------------------------------

START = (2, 0)           # Bottom-left corner
GOAL = (0, 3)            # +10 Reward
TRAP = (1, 3)            # -10 Reward
WALL = (1, 1)            # Blocked Cell


# -------------------------------------------------
# Reward Matrix
# -------------------------------------------------

R = np.full((ROWS, COLS), living_reward)

R[GOAL] = 10
R[TRAP] = -10


# -------------------------------------------------
# Available Actions
# -------------------------------------------------

actions = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}


# -------------------------------------------------
# Arrow Symbols
# -------------------------------------------------

arrows = {
    "UP": "\u2191",
    "DOWN": "\u2193",
    "LEFT": "\u2190",
    "RIGHT": "\u2192"
}


# -------------------------------------------------
# Find Next State
# -------------------------------------------------

def next_state(state, action):

    r, c = state
    dr, dc = action

    nr = r + dr
    nc = c + dc

    # Check boundary
    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
        return state

    # Check wall
    if (nr, nc) == WALL:
        return state

    return (nr, nc)


# -------------------------------------------------
# Print Grid
# -------------------------------------------------

def print_grid():

    print("\nGrid World\n")

    for r in range(ROWS):

        for c in range(COLS):

            if (r, c) == GOAL:
                print("+10", end="\t")

            elif (r, c) == TRAP:
                print("-10", end="\t")

            elif (r, c) == WALL:
                print("XXXX", end="\t")

            elif (r, c) == START:
                print("START", end="\t")

            else:
                print("*", end="\t")

        print()


# -------------------------------------------------
# Print Utility Matrix
# -------------------------------------------------

def print_utility(U):

    print()

    for r in range(ROWS):

        for c in range(COLS):

            if (r, c) == WALL:
                print("WALL", end="\t")

            else:
                print(f"{U[r, c]:6.2f}", end="\t")

        print()

    print()


# -------------------------------------------------
# Value Iteration
# -------------------------------------------------

def value_iteration():

    U = np.zeros((ROWS, COLS))

    U[GOAL] = 10
    U[TRAP] = -10

    iteration = 0

    while True:

        iteration += 1

        newU = U.copy()

        delta = 0

        for r in range(ROWS):

            for c in range(COLS):

                state = (r, c)

                # Skip special states
                if state in [GOAL, TRAP, WALL]:
                    continue

                values = []

                for move in actions.values():

                    ns = next_state(state, move)

                    values.append(U[ns])

                newU[state] = R[state] + gamma * max(values)

                delta = max(
                    delta,
                    abs(newU[state] - U[state])
                )

        U = newU

        print("\nIteration", iteration)

        print_utility(U)

        if delta < threshold:
            break

    policy = extract_policy(U)

    return U, policy


# -------------------------------------------------
# Extract Optimal Policy
# -------------------------------------------------

def extract_policy(U):

    policy = np.empty((ROWS, COLS), dtype=object)

    for r in range(ROWS):

        for c in range(COLS):

            state = (r, c)

            if state == WALL:

                policy[r, c] = "####"

            elif state == GOAL:

                policy[r, c] = "+10"

            elif state == TRAP:

                policy[r, c] = "-10"

            else:

                best_action = None
                best_value = -999

                for name, move in actions.items():

                    ns = next_state(state, move)

                    if U[ns] > best_value:

                        best_value = U[ns]
                        best_action = name

                policy[r, c] = arrows[best_action]

    return policy


# -------------------------------------------------
# Print Policy
# -------------------------------------------------

def print_policy(policy):

    print()

    for row in policy:

        for item in row:

            print(f"{item:^5}", end="\t")

        print()

    print()


# -------------------------------------------------
# Policy Iteration
# -------------------------------------------------

def policy_iteration():

    U = np.zeros((ROWS, COLS))

    U[GOAL] = 10
    U[TRAP] = -10

    # Initial policy: UP for all states
    policy = np.full(
        (ROWS, COLS),
        "UP",
        dtype=object
    )

    stable = False

    while not stable:

        # -------------------
        # Policy Evaluation
        # -------------------

        for _ in range(50):

            for r in range(ROWS):

                for c in range(COLS):

                    state = (r, c)

                    if state in [GOAL, TRAP, WALL]:
                        continue

                    action = actions[policy[r, c]]

                    ns = next_state(state, action)

                    U[state] = R[state] + gamma * U[ns]

        # -------------------
        # Policy Improvement
        # -------------------

        stable = True

        for r in range(ROWS):

            for c in range(COLS):

                state = (r, c)

                if state in [GOAL, TRAP, WALL]:
                    continue

                old_action = policy[r, c]

                best_action = old_action
                best_value = -999

                for name, move in actions.items():

                    ns = next_state(state, move)

                    if U[ns] > best_value:

                        best_value = U[ns]
                        best_action = name

                policy[r, c] = best_action

                if old_action != best_action:
                    stable = False

    # Convert action names to arrows
    display_policy = np.empty(
        (ROWS, COLS),
        dtype=object
    )

    for r in range(ROWS):

        for c in range(COLS):

            state = (r, c)

            if state == WALL:
                display_policy[r, c] = "####"

            elif state == GOAL:
                display_policy[r, c] = "+10"

            elif state == TRAP:
                display_policy[r, c] = "-10"

            else:
                display_policy[r, c] = arrows[policy[r, c]]

    return U, display_policy


# -------------------------------------------------
# Main Function
# -------------------------------------------------

def main():

    print("========== INITIAL GRID ==========")

    print_grid()

    # -------------------------
    # Value Iteration
    # -------------------------

    print("\n========== VALUE ITERATION ==========")

    U1, P1 = value_iteration()

    print("\nFinal Utilities")

    print_utility(U1)

    print("\nOptimal Policy")

    print_policy(P1)

    # -------------------------
    # Policy Iteration
    # -------------------------

    print("\n========== POLICY ITERATION ==========")

    U2, P2 = policy_iteration()

    print("\nFinal Utilities")

    print_utility(U2)

    print("\nOptimal Policy")

    print_policy(P2)


# -------------------------------------------------
# Run Program
# -------------------------------------------------

if __name__ == "__main__":
    main()

```

## OUTPUT
```
========== INITIAL GRID ==========

Grid World

*	*	*	+10	
*	XXXX	*	-10	
START	*	*	*	

========== VALUE ITERATION ==========

Iteration 1

 -1.00	 -1.00	  8.00	 10.00	
 -1.00	WALL	 -1.00	-10.00	
 -1.00	 -1.00	 -1.00	 -1.00	


Iteration 2

 -1.90	  6.20	  8.00	 10.00	
 -1.90	WALL	  6.20	-10.00	
 -1.90	 -1.90	 -1.90	 -1.90	


Iteration 3

  4.58	  6.20	  8.00	 10.00	
 -2.71	WALL	  6.20	-10.00	
 -2.71	 -2.71	  4.58	 -2.71	


Iteration 4

  4.58	  6.20	  8.00	 10.00	
  3.12	WALL	  6.20	-10.00	
 -3.44	  3.12	  4.58	  3.12	


Iteration 5

  4.58	  6.20	  8.00	 10.00	
  3.12	WALL	  6.20	-10.00	
  1.81	  3.12	  4.58	  3.12	


Iteration 6

  4.58	  6.20	  8.00	 10.00	
  3.12	WALL	  6.20	-10.00	
  1.81	  3.12	  4.58	  3.12	


Final Utilities

  4.58	  6.20	  8.00	 10.00	
  3.12	WALL	  6.20	-10.00	
  1.81	  3.12	  4.58	  3.12	


Optimal Policy

  →  	  →  	  →  	 +10 	
  ↑  	#### 	  ↑  	 -10 	
  ↑  	  →  	  ↑  	  ←  	


========== POLICY ITERATION ==========

Final Utilities

  4.58	  6.20	  8.00	 10.00	
  3.12	WALL	  6.20	-10.00	
  1.81	  3.12	  4.58	  3.12	


Optimal Policy

  →  	  →  	  →  	 +10 	
  ↑  	#### 	  ↑  	 -10 	
  ↑  	  →  	  ↑  	  ←  	


```

## RESULT:
Thus to Solve a Stochastic Grid-World Markov Decision Process Using Value Iteration and Policy Iteration is executed and displayed successfully.
