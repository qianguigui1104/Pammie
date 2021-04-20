"""Implements an example of using  the teaching algorithms
with the 3x3 reach domain.

Teaching algorithms:
1) Importance value
2) Huang et al
3) MCTS
"""

import numpy as np
import os

import policy_tutor.tutor_ai.importance_advising as importance
import policy_tutor.tutor_ai.huang as huang
import policy_tutor.tutor_ai.teacher_mcts as mcts

import policy_tutor.domains.reach_3x3_grid.env as env_lib
import policy_tutor.domains.reach_3x3_grid.robot as robot_lib
import policy_tutor.mdps.mdp as mdp_lib
import policy_tutor.utils.cache_utils as cache_lib


def main():
  env = env_lib.GridEnv()
  path = "../domains/reach_3x3_grid/data"
  policy = np.zeros((9, 5))

  # Read robot policy
  robot_policy = cache_lib.read_from_memmap(
      policy, os.path.join(path, "np_robot_policy.dat"))

  # Read candidate student policies
  student_policy_alpha = cache_lib.read_from_memmap(
      policy, os.path.join(path, "np_student_alpha_policy.dat"))

  student_policy_beta = cache_lib.read_from_memmap(
      policy, os.path.join(path, "np_student_beta_policy.dat"))

  candidate_policies = np.array([student_policy_alpha, student_policy_beta])

  # Define prior belief over policies
  prior_belief = np.full(candidate_policies.shape[0],
                         1 / candidate_policies.shape[0])

  # Use the importance advising algorithm
  demonstrations = importance_demonstrations(env, robot_policy)

  save_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "data/reach_3x3/importance_demonstrations.dat",
  )
  cache_lib.save_to_memmap(demonstrations, save_path)

  # Use the Huang et al approach
  demonstrations = huang_approach(prior_belief, env, robot_policy,
                                  candidate_policies)
  save_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "data/reach_3x3/huang_demonstrations.dat",
  )
  cache_lib.save_to_memmap(demonstrations, save_path)

  # Use the mcts approach
  demonstrations = mcts_approach(env, robot_policy, candidate_policies,
                                 prior_belief)

  save_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "data/reach_3x3/mcts_demonstrations.dat",
  )
  cache_lib.save_to_memmap(demonstrations, save_path)


def importance_demonstrations(env, robot_policy):
  """Selects demonstrations for the 3x3 grid domain using importance advising.
  """
  transition_model = env.np_transition_model
  robot = robot_lib.RobotModel()
  robot_reward_model = robot.np_reward_model

  robot_q_value = mdp_lib.q_value_from_policy(
      policy=robot_policy,
      transition_model=transition_model,
      reward_model=robot_reward_model,
      discount_factor=0.95,
      max_iteration=1000,
  )

  # threshold: Desired importance threshold.
  threshold = 180

  # budget: maximum number of demonstrations that can be selected.
  budget = 10

  # Generates demonstrations
  demonstrations = importance.importance_advising(robot_q_value, threshold,
                                                  budget)

  return demonstrations


def huang_approach(prior_belief, env, agent_policy, candidate_policies):
  """Selects demonstrations for the 3x3 grid domain using Huang et al approach.
  """
  # budget: maximum number of demonstrations allowed.
  budget = 3

  # trajectory_length: the length of the trajectory shown to the student.
  trajectory_length = 4

  # threshold: threshold for using the deterministic effect.
  threshold = 2

  # Generates demonstrations
  demonstrations = huang.select_examples(prior_belief, env, agent_policy,
                                         candidate_policies, budget,
                                         trajectory_length, threshold)

  return demonstrations


def mcts_approach(env, robot_policy, student_policies, prior_belief):
  """Selects demonstrations for the 3x3 grid domain using MCTS approach.
  """
  demonstration_budget = 3
  simulations_number = 200
  trajectory_length = 3

  demonstrations = mcts.select_demonstrations(env, robot_policy,
                                              student_policies, prior_belief,
                                              demonstration_budget,
                                              simulations_number,
                                              trajectory_length)

  return demonstrations


if __name__ == '__main__':
  main()
