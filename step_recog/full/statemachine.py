# Correcting indentation for docstrings and rerunning the module code with tests
import numpy as np
import random

STATE_UNOBSERVED = 0
STATE_CURRENT = 1
STATE_DONE = 2

class ProcedureStateMachine:
  """
  A state machine to process a sequence of probabilities representing steps in a procedure
  and determine the current state of each step as unobserved, current, or observed.

  Attributes:
      current_state (numpy.ndarray): The current state of each step in the procedure.
  """
  def __init__(self, num_steps):
    """
    Initializes the ProcedureStateMachine with a given number of steps.
    
    Args:
        num_steps (int): The number of steps in the procedure (excluding the 'no step').
    """
    self.num_steps = num_steps
    self.reset()

  def reset(self):
    self.current_state = np.zeros(self.num_steps, dtype=int) ## STATE_UNOBSERVED

  def process_timestep(self, probabilities):
    """
    Processes a single timestep, updating the current state based on the probabilities.
    
    Args:
        probabilities (numpy.ndarray): Probabilities for each step including 'no step' at the last index.
    """
    step_probabilities = probabilities[:-1]
    max_prob = np.max(step_probabilities)
    max_prob_no_step = probabilities[-1]
    
    max_indices = np.where(step_probabilities == max_prob)[0]
    
##    print("max_prob", max_prob, "max_prob_no_step", max_prob_no_step, "state", self.current_state)

    if max_prob_no_step >= max_prob:
      if np.sum(self.current_state == STATE_CURRENT) == 1 and np.sum(self.current_state == STATE_DONE) == len(self.current_state) - 1:
        self.current_state[:] = STATE_DONE
      return

    chosen_index = random.choice(max_indices) if len(max_indices) > 1 else max_indices[0]
    
    if self.current_state[chosen_index] != STATE_CURRENT:
      self.current_state[self.current_state == STATE_CURRENT] = STATE_DONE
      self.current_state[chosen_index] = STATE_CURRENT

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