from consts import *
from network import PolicyNetwork, ValueNetwork
import numpy as np
from queue import Queue, Empty, Full
import tensorflow as tf
import threading
import util

flags = tf.app.flags
flags.DEFINE_string('run_dir', 'latest', 'Run directory to load networks from')
flags.DEFINE_string('prior_network', 'network-1', 'Name of prior network')
flags.DEFINE_string('rollout_network', 'policy', 'Name of policy network')
flags.DEFINE_string('value_network', 'value', 'Name of value network')
flags.DEFINE_integer('mcts_threads', 1, 'Number of MCTS threads to run')
flags.DEFINE_integer('prior_threads', 2, 'Number of prior threads to run')
flags.DEFINE_integer('rollout_threads', 4, 'Number of rollout threads to run')
flags.DEFINE_integer('value_threads', 2, 'Number of value threads to run')
flags.DEFINE_float('prior_temperature', 10.0,
                   'Softmax temperature in prior network')
flags.DEFINE_float('rollout_temperature', 1.0,
                   'Softmax temperature in rollout network')
flags.DEFINE_float('exploration_rate', 5.0,
                   'Exploration rate to encourage visiting less seen nodes')
flags.DEFINE_float('lambda_mixing', 0.5,
                   'Proportion of node value from rollouts vs evaluations')
flags.DEFINE_float(
    'virtual_loss', 3.0,
    'Virtual loss applied to discourage visiting duplicate nodes')
flags.DEFINE_integer('expansion_threshold', 10,
                     'Number of times a node is visited before expansion')
flags.DEFINE_float('timeout', 3, 'Seconds to run MCTS for')


class Alpha4(object):
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

  def __init__(self, config):
    self.config = config

    # Create session
    self.session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(
        allow_growth=True)))

    # Create networks
    self.prior_network = PolicyNetwork(config.prior_network)
    self.rollout_network = PolicyNetwork(config.rollout_network)
    self.value_network = ValueNetwork(config.value_network)

    # Load networks from checkpoints
    run_dir = util.run_directory(config)
    util.restore_network_or_fail(self.session, run_dir, self.prior_network)
    util.restore_network_or_fail(self.session, run_dir, self.rollout_network)
    util.restore_network_or_fail(self.session, run_dir, self.value_network)

    # Create queues
    self.prior_queue = AllQueue()
    self.rollout_queue = AllQueue(maxsize=16)
    self.value_queue = AllQueue(maxsize=16)

    self.new_game()

  def new_game(self):
    self.transpositions = {}

  def best_move(self, position, timeout=None):
    self.mcts_threads_running = self.queue_threads_running = True
    timer = self.start_timer(timeout or self.config.timeout)
    self.prune_transpositions(position)
    root_node = self.expand_root_node(position)
    queue_threads = self.start_queue_threads()
    mcts_threads = self.start_mcts_threads(root_node)
    self.wait_for_threads(timer, queue_threads, mcts_threads)
    return self.most_played_move(position, root_node)

  def start_timer(self, timeout):
    timer = threading.Timer(timeout, self.stop)
    timer.start()
    return timer

  def stop(self):
    self.mcts_threads_running = False

  def prune_transpositions(self, position):
    self.transpositions = {
        cached_position: node
        for cached_position, node in self.transpositions.items()
        if position.is_ancestor(cached_position)
    }

  def expand_root_node(self, position):
    root_node = self.get_or_create_node(position)

    # Expand 3 levels down from root node
    self.expand(root_node)
    for child in root_node.children:
      self.expand(child)
      for grandchild in child.children:
        self.expand(grandchild)

    return root_node

  def start_queue_threads(self):
    queue_threads = []

    for i in range(self.config.prior_threads):
      prior_thread = threading.Thread(
          target=self.prior_thread, name='prior-%d' % i)
      prior_thread.start()
      queue_threads.append(prior_thread)

    for _ in range(self.config.rollout_threads):
      rollout_thread = threading.Thread(
          target=self.rollout_thread, name='rollout-%d' % i)
      rollout_thread.start()
      queue_threads.append(rollout_thread)

    for _ in range(self.config.value_threads):
      value_thread = threading.Thread(
          target=self.value_thread, name='value-%d' % i)
      value_thread.start()
      queue_threads.append(value_thread)

    return queue_threads

  def start_mcts_threads(self, root_node):
    mcts_threads = []

    for i in range(self.config.mcts_threads):
      mcts_thread = threading.Thread(
          target=self.mcts_thread, args=[root_node], name='mcts-%d' % i)
      mcts_thread.start()
      mcts_threads.append(mcts_thread)

    return mcts_threads

  def wait_for_threads(self, timer, queue_threads, mcts_threads):
    # MCTS threads are stopped first to stop filling up the queues
    for mcts_thread in mcts_threads:
      mcts_thread.join()

    # Now wait for queue threads to clear queues
    self.queue_threads_running = False
    for queue_thread in queue_threads:
      queue_thread.join()

    # Cancel timer in case stop was called by something else
    timer.cancel()

  def most_played_move(self, position, root_node):
    print(root_node.value)
    print(np.round(root_node.priors, 4))
    print(root_node.rollout_counts.astype(np.int))
    print(root_node.rollout_counts.sum())
    print([c.value for c in root_node.children])
    print()

    columns = position.legal_columns()
    rollouts = root_node.rollout_counts
    return columns[np.argmax(rollouts)]

  # Monte-Carlo Tree Search
  def mcts_thread(self, root_node):
    while self.mcts_threads_running:
      self.mcts(root_node)

  def mcts(self, root_node):
    node, node_path = self.select(root_node)
    self.evaluate(node, node_path)
    self.rollout(node, node_path)

  # Selection
  def select(self, node):
    node.apply_virtual_loss()
    node_path = []
    while node.children:
      index, child = node.max_action_child()
      node_path.append((node, index))
      node = child
      if node.rollout_count >= self.config.expansion_threshold:
        self.expand(node)
      node.apply_virtual_loss()

    return node, node_path

  # Expansion
  def expand(self, node):
    if node.children or node.terminal: return

    node.expand(
        [self.get_or_create_node(child) for child in node.position.children()])

    self.add_to_queue(self.prior_queue, node)

  def get_or_create_node(self, position):
    if position not in self.transpositions:
      self.transpositions[position] = Node(position, self.config)

    return self.transpositions[position]

  def prior_thread(self):
    for nodes in self.queued_items(self.prior_queue):
      nodes = list(set(nodes))
      priors, legal_moves = self.priors([node.position for node in nodes])
      for node, priors, legal_moves in zip(nodes, priors, legal_moves):
        node.update_priors(priors[legal_moves.reshape(TOTAL_DISKS)])

  def priors(self, positions):
    turns = [position.turn for position in positions]
    disks = [position.disks for position in positions]
    empty = [position.empty for position in positions]
    legal_moves = [position.legal_moves for position in positions]
    threats = [position.threats for position in positions]

    priors = self.session.run(self.prior_network.policy, {
        self.prior_network.turn: turns,
        self.prior_network.disks: disks,
        self.prior_network.empty: empty,
        self.prior_network.legal_moves: legal_moves,
        self.prior_network.threats: threats,
        self.prior_network.temperature: self.config.prior_temperature
    })

    return priors, legal_moves

  # Evaluation
  def evaluate(self, node, node_path):
    if node.evaluated:
      evaluation = node.evaluation
      for parent, _ in node_path:
        parent.update_leaf_evaluation(evaluation)
    else:
      self.add_to_queue(self.value_queue, (node, node_path))

  def value_thread(self):
    for evaluations in self.queued_items(self.value_queue):
      positions = [node.position for (node, _) in evaluations]
      values = self.values(positions)

      for (node, node_path), value in zip(evaluations, values):
        node.set_evaluation(value)
        for parent, _ in node_path:
          parent.update_leaf_evaluation(value)

  def values(self, positions):
    if not positions: return []

    turns = [position.turn for position in positions]
    disks = [position.disks for position in positions]
    empty = [position.empty for position in positions]
    legal_moves = [position.legal_moves for position in positions]
    threats = [position.threats for position in positions]

    return self.session.run(self.value_network.value, {
        self.value_network.turn: turns,
        self.value_network.disks: disks,
        self.value_network.empty: empty,
        self.value_network.legal_moves: legal_moves,
        self.value_network.threats: threats
    })

  # Rollouts
  def rollout(self, node, node_path):
    self.add_to_queue(self.rollout_queue, (node, node_path))

  def rollout_thread(self):
    for rollouts in self.queued_items(self.rollout_queue):
      self.run_rollouts(rollouts)

  def run_rollouts(self, rollouts):
    positions = [node.position for (node, _) in rollouts]
    while rollouts:
      moves = self.rollout_moves(positions)
      new_positions, new_rollouts = [], []

      for position, rollout, move in zip(positions, rollouts, moves):
        position = position.move(move)
        if position.gameover():
          result = position.result
          node, node_path = rollout
          node.update_rollout(result)
          for parent, child_index in node_path:
            parent.update_rollout(result, child_index)
        else:
          new_positions.append(position)
          new_rollouts.append(rollout)

      positions = new_positions
      rollouts = new_rollouts

  def rollout_moves(self, positions):
    if not positions: return []

    turns = [position.turn for position in positions]
    disks = [position.disks for position in positions]
    empty = [position.empty for position in positions]
    legal_moves = [position.legal_moves for position in positions]
    threats = [position.threats for position in positions]

    return self.session.run(self.rollout_network.sample_move, {
        self.rollout_network.turn: turns,
        self.rollout_network.disks: disks,
        self.rollout_network.empty: empty,
        self.rollout_network.legal_moves: legal_moves,
        self.rollout_network.threats: threats,
        self.rollout_network.temperature: self.config.rollout_temperature
    })

  # Queue utils
  def add_to_queue(self, queue, item):
    while self.mcts_threads_running:
      try:
        queue.put(item, timeout=0.01)
        return
      except Full:
        pass

  def queued_items(self, queue):
    while self.queue_threads_running or not queue.empty:
      try:
        yield queue.get(timeout=0.01)
      except Empty:
        pass


class Node(object):
  __slots__ = [
      'position', 'config', 'terminal', 'evaluated', 'value', 'children',
      'leaf_evaluation_total', 'leaf_evaluation_count', 'rollout_total',
      'rollout_count', 'priors', 'rollout_counts', 'evaluation'
  ]

  def __init__(self, position, config):
    self.position = position
    self.config = config
    self.terminal = position.gameover()
    if self.terminal:
      self.set_evaluation(position.result)
    else:
      self.evaluated = False
      self.value = 0
    self.children = []

    self.leaf_evaluation_total = 0
    self.leaf_evaluation_count = EPSILON  # Avoid divide by 0
    self.rollout_total = 0
    self.rollout_count = EPSILON  # Avoid divide by 0

  def expand(self, children):
    num_children = len(children)
    # Default to uniform priors
    self.priors = np.tile(1 / num_children, num_children)
    self.rollout_counts = np.tile(EPSILON, num_children)  # Avoid divide by 0
    self.children = children

  def apply_virtual_loss(self):
    self.rollout_total -= self.config.virtual_loss
    self.rollout_count += self.config.virtual_loss

  def update_priors(self, priors):
    self.priors = priors

  def update_rollout(self, result, child_index=None):
    self.rollout_total += result + self.config.virtual_loss
    self.rollout_count += 1 - self.config.virtual_loss
    if child_index is not None:
      self.rollout_counts[child_index] += 1
    self.update_value()

  def update_leaf_evaluation(self, value):
    self.leaf_evaluation_total += value
    self.leaf_evaluation_count += 1
    self.update_value()

  def update_value(self):
    rollout_values = self.rollout_total / self.rollout_count
    leaf_values = self.leaf_evaluation_total / self.leaf_evaluation_count
    values = (self.config.lambda_mixing * rollout_values +
              (1 - self.config.lambda_mixing) * leaf_values)
    self.value = values * -util.turn_win(self.position.turn)

  def max_action_child(self):
    values = np.array([child.value for child in self.children])
    score = values + self.config.exploration_rate * self.exploration_bonus()
    # Add noise to fairly break ties caused by uniform priors
    score += np.random.uniform(low=0, high=0.0001, size=len(self.children))
    index = np.argmax(score)
    return index, self.children[index]

  def exploration_bonus(self):
    return (self.priors * np.sqrt(self.rollout_counts.sum()) /
            self.rollout_counts)

  def set_evaluation(self, value):
    self.evaluation = value
    self.value = value * -util.turn_win(self.position.turn)
    self.evaluated = True


class AllQueue(Queue):
  """Queue that returns all items on `get`"""

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
  alpha4 = Alpha4(flags.FLAGS)
  from position import Position
  position = Position().move(1).move(4).move(3).move(4).move(3).move(4).move(
      3).move(3)
  print(position)
  result = alpha4.best_move(position)
  print(result)
