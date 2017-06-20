from consts import *
from network import PolicyNetwork, ValueNetwork
import numpy as np
from players import PolicyPlayer
import queue
import tensorflow as tf
import threading
import util


class Alpha4(object):
  def __init__(self, network_name, run_dir):

    # self.network = PolicyPlayer(PolicyNetwork(network_name), self.session)
    self.alpha_search = AlphaSearch(run_dir)
    # self.session.run(tf.global_variables_initializer())

  def play(self, position):
    return self.alpha_search.best_move(position)
    # print(self.position.legal_moves())

    # return self.network.play([position])[0]

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

  def __init__(self, run_dir):
    # self.rollouts = 0
    # self.max_rollouts = 2000
    # self.max_rollouts = 100
    # self.value_queue = []
    # self.virtual_loss = 3
    self.expansion_threshold = 3

    self.session = tf.Session()
    # self.session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(
    #     allow_growth=True)))
    self.prior_network = PolicyNetwork('network-1')
    self.rollout_network = PolicyNetwork('policy')
    self.value_network = ValueNetwork('value')
    # self.prior_thread = PriorThread(session, self.prior_network)
    # self.rollout_thread = RolloutThread(session, self.rollout_network)
    print(run_dir)
    util.try_restore(self.session, run_dir, self.prior_network)
    util.try_restore(self.session, run_dir, self.rollout_network)
    util.try_restore(self.session, run_dir, self.value_network)

  # def stop(self):
  #   self.running = False
  #   self.prior_thread.stop()
  #   # self.rollout_thread.stop()

  def best_move(self, position):
    self.running = True
    self.node_cache = {}
    self.prior_queue = AllQueue()
    self.rollout_queue = AllQueue(maxsize=32)
    self.evaluation_queue = AllQueue(maxsize=32)

    threads = []
    for _ in range(2):
      prior_thread = PriorThread(self.prior_queue, self.session,
                                 self.prior_network)
      threads.append(prior_thread)

    for _ in range(2):
      rollout_thread = RolloutThread(self.rollout_queue, self.session,
                                     self.rollout_network)
      threads.append(rollout_thread)

    for _ in range(2):
      evaluation_thread = EvaluationThread(self.evaluation_queue, self.session,
                                           self.value_network)
      threads.append(evaluation_thread)

    for thread in threads:
      thread.start()

    def stop():
      self.running = False
      for thread in threads:
        thread.stop()

    stop_timer = threading.Timer(1.0, stop)
    stop_timer.start()

    root_node = self.get_or_create_node(position)
    self.expand(root_node)
    for node in root_node.children:
      self.expand(node)

    self.mcts_thread(root_node)

    # Return column with greatest number of rollouts
    print(np.round(root_node.priors, 4))
    print(root_node.rollout_count.astype(np.int))
    print(root_node.rollout_count.sum())
    print()
    return position.legal_columns()[np.argmax(root_node.rollout_count)]

  def mcts_thread(self, root_node):
    while self.running:
      self.mcts(root_node)

  def mcts(self, root_node):
    # Selection
    node = root_node
    node_path = []
    while not node.is_leaf:
      index, child = node.max_action_child()
      node_path.append((node, index))
      node = child
      node.visits += 1
      if not node.terminal and node.visits >= self.expansion_threshold:
        self.expand(node)

    # Evaluation
    if node.evaluated:
      value = node.value
      for parent, child_index in node_path:
        parent.leaf_evaluation_total[child_index] += value
        parent.leaf_evaluation_count[child_index] += 1
    else:
      self.add_to_queue(self.evaluation_queue, (node, node_path))


    # Rollout
    self.add_to_queue(self.rollout_queue, (node.position, node_path))

  def expand(self, node):
    if node.terminal: return

    children = [
        self.get_or_create_node(child) for child in node.position.children()
    ]
    node.expand(children)
    # self.prior_thread.add_to_queue(node)
    self.add_to_queue(self.prior_queue, node)

  def add_to_queue(self, queue, item):
    while self.running:
      try:
        queue.put(item, timeout=0.1)
        return
      except:
        pass

  def get_or_create_node(self, position):
    if position not in self.node_cache:
      self.node_cache[position] = Node(position)

    return self.node_cache[position]

  def evaluate(self, position):
    [value] = self.session.run(self.value_network.value, {
        self.value_network.turn: [position.turn],
        self.value_network.disks: [position.disks],
        self.value_network.empty: [position.empty],
        self.value_network.legal_moves: [position.legal_moves],
        self.value_network.threats: [position.threats]
    })
    return value


class PriorThread(threading.Thread):
  def __init__(self, queue, session, prior_network):
    super(PriorThread, self).__init__()
    self._session = session
    self._prior_network = prior_network
    self._queue = queue
    self._running = False

  def stop(self):
    self._running = False

  def run(self):
    self._running = True
    while self._running:
      try:
        nodes = list(set(self._queue.get(timeout=0.1)))
      except:
        nodes = []

      priors, legal_moves = self.priors([node.position for node in nodes])

      for node, priors, legal_moves in zip(nodes, priors, legal_moves):
        node.update_priors(priors[legal_moves.reshape(TOTAL_DISKS)])
        self._running = False

  def priors(self, positions):
    if not positions: return [], []

    turns = [position.turn for position in positions]
    disks = [position.disks for position in positions]
    empty = [position.empty for position in positions]
    legal_moves = [position.legal_moves for position in positions]
    threats = [position.threats for position in positions]

    priors = self._session.run(self._prior_network.policy, {
        self._prior_network.turn: turns,
        self._prior_network.disks: disks,
        self._prior_network.empty: empty,
        self._prior_network.legal_moves: legal_moves,
        self._prior_network.threats: threats,
        self._prior_network.temperature: 10.0
    })

    return priors, legal_moves


class RolloutThread(threading.Thread):
  def __init__(self, queue, session, rollout_network):
    super(RolloutThread, self).__init__()
    self._session = session
    self._rollout_network = rollout_network
    self._queue = queue
    self._running = False

  def stop(self):
    self._running = False

  def run(self):
    self._running = True
    rollouts = []
    while self._running:
      if not rollouts:
        rollouts = self._queue.get()

      positions = [position for (position, _) in rollouts]
      moves = self.sample_moves(positions)
      new_rollouts = []

      for (position, node_path), move in zip(rollouts, moves):
        position = position.move(move)
        if position.gameover():
          result = position.result
          for parent, child_index in node_path:
            parent.rollout_total[child_index] += result
            parent.rollout_count[child_index] += 1
        else:
          new_rollouts.append((position, node_path))

      rollouts = new_rollouts

  def sample_moves(self, positions):
    if not positions: return []

    turns = [position.turn for position in positions]
    disks = [position.disks for position in positions]
    empty = [position.empty for position in positions]
    legal_moves = [position.legal_moves for position in positions]
    threats = [position.threats for position in positions]

    return self._session.run(self._rollout_network.sample_move, {
        self._rollout_network.turn: turns,
        self._rollout_network.disks: disks,
        self._rollout_network.empty: empty,
        self._rollout_network.legal_moves: legal_moves,
        self._rollout_network.threats: threats,
        self._rollout_network.temperature: 5.0
    })


class EvaluationThread(threading.Thread):
  def __init__(self, queue, session, value_network):
    super(EvaluationThread, self).__init__()
    self._session = session
    self._value_network = value_network
    self._queue = queue
    self._running = False

  def stop(self):
    self._running = False

  def run(self):
    self._running = True
    while self._running:
      evaluations = self._queue.get()
      positions = [node.position for (node, _) in evaluations]
      values = self.values(positions)
      for (node, node_path), value in zip(evaluations, values):
        node.set_value(value)
        for parent, child_index in node_path:
          parent.leaf_evaluation_total[child_index] += value
          parent.leaf_evaluation_count[child_index] += 1

  def values(self, positions):
    if not positions: return []

    turns = [position.turn for position in positions]
    disks = [position.disks for position in positions]
    empty = [position.empty for position in positions]
    legal_moves = [position.legal_moves for position in positions]
    threats = [position.threats for position in positions]

    return self._session.run(self._value_network.value, {
        self._value_network.turn: turns,
        self._value_network.disks: disks,
        self._value_network.empty: empty,
        self._value_network.legal_moves: legal_moves,
        self._value_network.threats: threats
    })


EXPLORATION_RATE = 5
LAMBDA_MIXING = 0.5


class Node(object):
  def __init__(self, position):
    self.position = position
    self.terminal = position.gameover()
    if self.terminal:
      self.set_value(position.result)
    else:
      self.evaluated = False
    self.is_leaf = True
    self.visits = 0
    self.parents = []

  def expand(self, children):
    num_children = len(children)
    # Default to uniform priors
    self.priors = np.tile(1 / num_children, num_children)
    self.leaf_evaluation_total = np.zeros(num_children)
    self.leaf_evaluation_count = np.tile(EPSILON, num_children)
    self.rollout_total = np.zeros(num_children)
    self.rollout_count = np.tile(EPSILON, num_children)

    self.children = children
    for child in children:
      child.add_parent(self)

    self.is_leaf = False

  def add_parent(self, parent):
    self.parents.append(parent)

  def update_priors(self, priors):
    self.priors = priors

  def exploration_bonus(self):
    return (self.priors * np.sqrt(self.rollout_count.sum()) /
            (self.rollout_count + 1))

  def max_action_child(self):
    score = self.action_values() + EXPLORATION_RATE * self.exploration_bonus()
    index = np.argmax(score)
    return index, self.children[index]

  def action_values(self):
    leaf_values = self.leaf_evaluation_total / self.leaf_evaluation_count
    rollout_values = self.rollout_total / self.rollout_count
    action_values = (1 - LAMBDA_MIXING
                     ) * leaf_values + LAMBDA_MIXING * rollout_values
    return action_values * util.turn_win(self.position.turn)

  def set_value(self, value):
    self.value = value
    self.evaluated = True


class AllQueue(queue.Queue):
  def _init(self, maxsize):
    self.queue = []

  def _qsize(self):
    return len(self.queue)

  def _put(self, item):
    self.queue.append(item)

  def _get(self):
    result, self.queue = self.queue, []
    return result


if __name__ == '__main__':
  alpha4 = Alpha4('alpha4', 'runs/run_7')
  import cProfile as profile
  from position import Position
  position = Position().move(0).move(0).move(0).move(0).move(0).move(0)
  # profile.runctx('result = alpha4.play(position)',
  #                globals(), locals(), 'tmp.prof')
  result = alpha4.play(position)
  print(result)
  # print(alpha4.play(Position()))
