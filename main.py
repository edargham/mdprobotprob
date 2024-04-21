import mdptoolbox as mdpT
import numpy as np

def fill_transition_matrix(S, A, P, decay_rate=None):
  # The decay rate is not the markovian discount factor.
  # It is rather a bias term that the battery will discharge faster as it is used more.

  rate_current = 0.6
  rate_next = 0.4

  for s in range(S):
    P[A][s, s] = rate_current  # current state
    P[A][s, (s+1)%S] = rate_next  # next state
    if decay_rate is not None:
      rate_current -= decay_rate
      rate_next += decay_rate
  return P

def main():
  S = 10
  A = 3

  search = 0
  wait = 1
  recharge = 2

  gamma=0.9

  P = np.zeros((A, S, S))

  P = fill_transition_matrix(S, search, P, decay_rate=0.05)
  P = fill_transition_matrix(S, wait, P)
  
  for s in range(S):
    P[recharge][s, 0] = 1

  R = np.zeros((S, A))

  R[:,search] = [5 for _ in range(S-1)] + [5 * P[search][S-1, -1] + -3 * P[search][S-1, 0]]
  R[:,wait] = [2 for _ in range(S)] # //2)] + [7 for _ in range(S//2 -1)] + [2]
  R[:,recharge] = [-2 for _ in range(S//2)] + [0 for _ in range(S//2)]

  print('P=\n', P)
  print('R=\n', R)

  prob=mdpT.mdp.PolicyIteration(P,R,gamma)
  prob.run()

  print('Optimal policy:',prob.policy)
  print('Optimal values:',prob.V)

if __name__ == "__main__":
  main()