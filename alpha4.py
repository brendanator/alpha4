from consts import *
from network import Network
import numpy as np
from players import NetworkPlayer
import tensorflow as tf
import util


class Alpha4(object):
  def __init__(self, network_name, run_dir):
    self.session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(
        allow_growth=True)))
    self.network = NetworkPlayer(Network(network_name), self.session)
    self.network = RandomPlayer()
    util.restore(self.session, run_dir, 'policy')

  def play(self, position):
    search = AlphaSearch()
    # return search.evaluate(position)
    # print(self.position.legal_moves())

    return self.network.play([position])[0]

    # return np.random.choice(position.legal_columns())


class AlphaSearch(object):
  """
  ### Selection
  - Start at the root node and descend the tree until we reach a leaf node
  - Select the max action $a_t = \argmax_a [Q(s_t,a) + u(s_t, a)]$ where $u(s_t,a) = c_{puct}P(s,a) \frac{\sqrt{\sum_b N_r(s,b)}}{1 + N_r(s,a)}$ where $c_{puct}$ is a constant determining the level of exploration
  - This search control strategy initially prefers actions with high prior probability and low visit count, but asymptotically prefers actions with high action value
  ### Evaluation
  - The leaf is added to the queue for evaluation by the value network unless it already had been evaluated
  - The second rollout begins at the leaf node and continues until the end of the game using the rollout policy network and the final result $z_t$ is calculated
  ### Backup
  - At each in-tree step $t \leq L$ the rollout stats are updates as if it has lost $n_{vl}$ games. $N_r(s,a) \leftarrow N_r(s,a) + n_{vl}$, $W_r(s,a) \leftarrow W_r(s,a) - n_{vl}$. This virtual loss discourages other threads from search the same variation
  - At the ed of the simulation the rollout stats are updated with the outcome. $N_r(s,a) \leftarrow N_r(s,a) - n_{vl} + 1$, $W_r(s,a) \leftarrow W_r(s,a) + n_{vl} + z_t$
  - Asynchronously a separate backward pass updates the value statistics when the value network is complete
  - The overall evaluation is a weighted average of the Monte Carlo stats $Q(s,a) = (1-\lambda)\frac{W_v(s,a)}{N_v(s,a)} + \lambda\frac{W_r(s,a)}{N_r(s,a)}$
  - All updates are performed lock-free
  ### Expansion
  - When an edge visit count exceeds a threshold $n_{thr}$ the successor state is added to the search tree with initial values and becomes a leaf node
      - $N(s', a)=0$ - number of visits
      - $N_r(s', a)=0$ - number of rollout
      - $W(s', a)=0$ - win score by value
      - $W_r(s', a)=0$ - win score by rollout
      - $P(s',a)=p_{\sigma}^{\beta}(a|s')$ - prior action probabilities, where $p_{\sigma}^{\beta}$ is similar to the rollout policy network but with more features and $\beta$ is the softmax temperature
  - Then the state is added to 2 queues nt have the value and policy networks evaluated for the state. The prior probability is then updated with the policy network results which also uses the softmax temperature $\beta$
  o
  """

  def __init__(self):
    self.rollouts = 0
    self.max_rollouts = 20
    self.value_queue = []
    self.virtual_loss = 3

  def evaluate(self, position):
    return self.rollout_thread(TreeNode(position))

  def rollout_thread(self, root_node):
    while self.rollouts < self.max_rollouts:
      self.rollouts += 1

      # selection
      selection_path = []
      node = root_node
      while True:
        selection_path.append(node)
        node = node.max_action_child()
        if node.is_leaf:
          # if node.visits >= self.expansion_threshold:
          #   node = self.expand(node)
          # else:
          break

      # evaluation
      if not node.evaluated:
        self.value_queue.append((node, selection_path))
        if len(self.value_queue) == 16:
          value_nodes = [node for (node, _) in self.value_queue]
          values = self.node_values(value_nodes)
          for (value_node, path), value in zip(self.value_queue, values):
            value_node.evaluated = True
            for node in path:
              node.value += value
              node.visits += 1

      for parent in selection_path:
        parent.visits += self.virtual_loss
        parent.value -= self.virtual_loss

      rollout_value = self.rollout(node)

      # Backup
      for parent in selection_path:
        parent.visits += 1 - self.virtual_loss
        parent.value += self.virtual_loss + rollout_value

  def node_values(self, nodes):
    return [node.position.result or 0 for node in nodes]

  def rollout(self, node):
    position = node.position
    while not position.gameover:
      moves = position.legal_moves()
      position = position.move(np.random.choice(moves))

    return position.result

  def selection(self):
    node = self.root_node
    while not node.is_leaf:
      node = node.max_action_child()

  def expansion(self):
    pass

  def evaluation(self, node):
    if not node.evaluated:
      self.evaluation_queue.append(node)

    self.apply_virtual_loss(node.parents)
    self.rollout(node)
    self.rollout_backup(node)

  def value_backup(self):
    pass

  def rollout_backup(self):
    pass


class Node(object):
  pass


MAX_DEPTH = 3


class TreeNode(Node):
  def __init__(self, position, depth=0):
    self.position = position
    self.is_leaf = depth >= MAX_DEPTH
    self.evaluated = False
    self.exploration_preference = 5
    self.action_values = np.zeros(WIDTH)
    self.visits = np.zeros(WIDTH)
    self.total_visits = 0  # ?
    self.value = 0
    self.priors = np.zeros(WIDTH)
    if depth < MAX_DEPTH:
      self.children = [
          TreeNode(child, depth + 1) for child in position.children()
      ]

  def exploration_bonus(self):
    return self.exploration_preference * self.priors / (self.visits + 1)

  def max_action_child(self):
    if self.total_visits:
      score = self.action_values + self.exploration_bonus()
    else:
      score = self.priors

    return self.children[np.argmax(score)]


class LeafNode(Node):
  def __init__(self):
    self.is_leaf = True
    self.is_terminal = False
    self.visits = 0
    self.rollouts = 0
    self.value = 0
    self.rollout_wins = 0
    self.priors = np.zeros(WIDTH)


class RolloutNode(Node):
  pass


class Edges(object):
  def __init__(self):
    self.exploration_preference = 5
    self.action_values = np.zeros(WIDTH)
    self.visits = np.zeros(WIDTH)
    self.total_visits = 0  # ?
    self.priors = np.zeros(WIDTH)

  def exploration_bonus(self):
    return self.exploration_preference * self.priors / (self.visits + 1)

  def max_action_child(self):
    if self.total_visits:
      return np.argmax(self.action_values + self.exploration_bonus())
    else:
      return np.argmax(self.priors)


if __name__ == '__main__':
  from position import Position
  alpha4 = Alpha4(Position())
  print(alpha4.play())
