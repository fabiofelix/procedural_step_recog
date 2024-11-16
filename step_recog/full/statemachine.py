# Correcting indentation for docstrings and rerunning the module code with tests
import numpy as np
import random
from collections import deque
from enum import IntEnum

class State(IntEnum):
  UNOBSERVED = 0       
  CURRENT    = 1
  DONE       = 2  

class SmoothType(IntEnum):
  MEAN   = 0       
  MEDIAN = 1
  EXP    = 2    

class ProcedureStateMachine:
  """
  A state machine to process a sequence of probabilities representing steps in a procedure
  and determine the current state of each step as unobserved, current, or observed.

  Attributes:
      current_state (numpy.ndarray): The current state of each step in the procedure.
  """
  def __init__(self, num_steps, maxlen = 3):
    """
    Initializes the ProcedureStateMachine with a given number of steps.
    
    Args:
        num_steps (int): The number of steps in the procedure (excluding the 'no step').
        maxlen (int): Exponential moving average smoothes the predictions with a 'maxlen' window
                      It avoids noisy predictions changing the states.
                      Avoid using large 'maxlen' (e.g. >= 3)
    """
    self.num_steps = num_steps
    self.window_probs = deque(maxlen= 1 if maxlen is None else maxlen)
    self.ema_alpha = 0.5
    self.reset()

  def reset(self):
    self.current_state = np.zeros(self.num_steps, dtype=int) ## STATE_UNOBSERVED
    self.window_probs.clear()

  def smooth_probs(self, probabilities, type = SmoothType.MEDIAN):
    self.window_probs.append(probabilities)
    window_probs_aux = np.array(self.window_probs)
    move_avg = window_probs_aux[0] if window_probs_aux.shape[0] == 1 else np.median(window_probs_aux, axis = 0)

    if type == SmoothType.MEAN:
      move_avg = window_probs_aux[0] if window_probs_aux.shape[0] == 1 else window_probs_aux.mean(axis = 0)
    elif type == SmoothType.EXP:      
      move_avg = np.zeros(window_probs_aux.shape[1], dtype = window_probs_aux.dtype)

      for step, probs in enumerate(window_probs_aux.T):
        move_avg[step] = probs[0]

        for pb in probs[1:]:
          move_avg[step] = self.ema_alpha * pb + (1 - self.ema_alpha) * move_avg[step]

    #forces sum(move_avg) = 1.0
    move_avg    = move_avg / np.sum(move_avg)
    max_prob    = np.max(move_avg)
    max_indices = np.where(move_avg == max_prob)[0]

    return max_indices

  def process_timestep(self, probabilities):
    """
    Processes a single timestep, updating the current state based on the probabilities.
    
    Args:
        probabilities (numpy.ndarray): Probabilities for each step including 'no step' at the last index.
    """
    max_indices = self.smooth_probs(probabilities)

    chosen_index = random.choice(max_indices) if len(max_indices) > 1 else max_indices[0]
    
    if self.current_state[chosen_index] != State.CURRENT:
      self.current_state[self.current_state == State.CURRENT] = State.DONE
      self.current_state[chosen_index] = State.CURRENT

# Define the tests within the same environment
def run_tests():
    scenarios = {
        "Single step becomes current": np.array([[0.4, 0.3, 0.2, 0.1, 0.0]]).T,
        "Different step becomes current, previous observed": np.array([[0.4, 0.6, 0.0, 0.0, 0.0], [0.7, 0.5, 0.0, 0.0, 0.5]]).T,        
        "Special case: all but one observed, then no step max": np.array([
            [0.3, 0.1, 0.2, 0.1, 0.1],
            [0.1, 0.3, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.3, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.3, 0.1],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]).T,
        "Tie resolution by random choice": np.array([[0.25, 0.25, 0.25, 0.25, 0.0]]).T,        
    }

    results = {}
    for scenario_name, gru_output in scenarios.items():
        num_steps = gru_output.shape[0] - 1
        psm = ProcedureStateMachine(num_steps)
        for t in range(gru_output.shape[1]):
            psm.process_timestep(gru_output[:, t])
        results[scenario_name] = psm.current_state

    return results

# Run the tests
# test_results = run_tests()
# test_results      