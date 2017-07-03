from consts import *
from network import PolicyNetwork, ValueNetwork
import numpy as np
from queue import Queue, Empty, Full
import tensorflow as tf
import threading
import util

flags = tf.app.flags
flags.DEFINE_string('run_dir', 'latest', 'Run directory to load networks from')
flags.DEFINE_string('prior_network', 'policy', 'Name of prior network')
flags.DEFINE_string('rollout_network', 'policy', 'Name of policy network')
flags.DEFINE_string('value_network', 'value', 'Name of value network')
flags.DEFINE_integer('mcts_threads', 1, 'Number of MCTS threads to run')
flags.DEFINE_integer('prior_threads', 2, 'Number of prior threads to run')
flags.DEFINE_integer('rollout_threads', 4, 'Number of rollout threads to run')
flags.DEFINE_integer('value_threads', 2, 'Number of value threads to run')
flags.DEFINE_float('prior_temperature', 15.0,
                   'Softmax temperature in prior network')
flags.DEFINE_float('rollout_temperature', 5.0,
                   'Softmax temperature in rollout network')
flags.DEFINE_boolean(
    'use_symmetry', True,
    'Also feed horizontally flipped position into networks and take average')
flags.DEFINE_float('exploration_rate', 5.0,
                   'Exploration rate to encourage visiting less seen nodes')
flags.DEFINE_float(
    'virtual_loss', 3.0,
    'Virtual loss applied to discourage visiting duplicate nodes')
flags.DEFINE_float('rollout_proportion', 0.5,
                   'Proportion of node value from rollouts vs evaluations')
flags.DEFINE_integer('expansion_threshold', 20,
                     'Number of times a node is visited before expansion')
flags.DEFINE_float('timeout', 3.0, 'Seconds to run MCTS for')
flags.DEFINE_boolean('verbose', True, 'Print results of MCTS')
flags.DEFINE_string(
    'move_choice', 'rollouts',
    '[rollouts|value] - choose move by value or number of rollouts')


class Alpha4(object):
  def __init__(self, config):
    self.config = config

    # Create session
    self.session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(
        allow_growth=True)))

    # Create networks
    self.prior_network = PolicyNetwork(
        scope=config.prior_network,
        temperature=config.prior_temperature,
        use_symmetry=config.use_symmetry)

    self.rollout_network = PolicyNetwork(
        scope=config.rollout_network,
        temperature=config.rollout_temperature,
        reuse=config.prior_network == config.rollout_network,
        use_symmetry=config.use_symmetry)

    self.value_network = ValueNetwork(
        scope=config.value_network, use_symmetry=config.use_symmetry)

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
    if len(position.legal_columns()) == 1:
      return position.legal_columns()[0]
    elif position.counter_move is not None:
      return position.counter_move
    else:
      return self.run_mcts(position, timeout or self.config.timeout)

  def run_mcts(self, position, timeout):
    self.mcts_threads_running = self.queue_threads_running = True
    timer = self.start_timer(timeout)
    self.prune_transpositions(position)
    root_node = self.expand_root_node(position)
    queue_threads = self.start_queue_threads()
    mcts_threads = self.start_mcts_threads(root_node)
    self.wait_for_threads(timer, queue_threads, mcts_threads)
    return self.choose_best_move(position, root_node)

  def start_timer(self, timeout):
    timer = threading.Timer(timeout, self.stop)
    timer.start()
    return timer

  def stop(self):
    self.mcts_threads_running = False

  def prune_transpositions(self, position):
    # Prune transpositions
    self.transpositions = {
        cached_position: node
        for cached_position, node in self.transpositions.items()
        if position.is_ancestor(cached_position)
    }

    # Remove pruned parents from nodes
    all_nodes = set(self.transpositions.values())
    for node in all_nodes:
      node.parents &= all_nodes

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

  def choose_best_move(self, position, root_node):
    if self.config.verbose:
      np.set_printoptions(
          formatter={'float': '{: 0.4f}'.format,
                     'int': '{:7d}'.format})
      print(position)
      print('Priors        ', root_node.priors)
      print('Rollouts      ', root_node.rollout_counts.astype(np.int))
      print('Child values  ',
            np.array([child.value for child in root_node.children]))
      print('Value         ', '%.4f' % -root_node.value)
      print('Total rollouts', int(root_node.rollout_counts.sum()))
      print()

    if self.config.move_choice == 'rollouts':
      columns = position.legal_columns()
      rollouts = root_node.rollout_counts
      return columns[np.argmax(rollouts)]
    elif self.config.move_choice == 'value':
      columns = position.legal_columns()
      values = [child.value for child in root_node.children]
      return columns[np.argmax(values)]
    else:
      raise Exception('%s is not valid move_choice' % self.config.move_choice)

  # Monte-Carlo Tree Search
  def mcts_thread(self, root_node):
    while self.mcts_threads_running:
      self.mcts(root_node)

  def mcts(self, root_node):
    node, selection = self.select(root_node)
    self.evaluate(node, selection)
    self.rollout(node, selection)

  # Selection
  def select(self, node):
    selection = []
    while node.children:
      index, child = node.max_value_child()
      selection.append((node, index))
      node = child
      if node.rollout_count >= self.config.expansion_threshold:
        self.expand(node)

    node.backup_virtual_loss(selection)

    return node, selection

  # Expansion
  def expand(self, node):
    if node.children or node.terminal: return

    if node.position.counter_move is not None:
      node.expand([
          self.get_or_create_node(
              node.position.move(node.position.counter_move))
      ])
    else:
      node.expand([
          self.get_or_create_node(child) for child in node.position.children()
      ])
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
        priors = priors.reshape(HEIGHT, WIDTH).sum(axis=0)
        priors = priors[node.position.legal_columns()]
        node.update_priors(priors)

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
        self.prior_network.threats: threats
    })

    return priors, legal_moves

  # Evaluation
  def evaluate(self, node, selection):
    if node.evaluated:
      node.backup_evaluation_result(node.evaluation, selection)
    else:
      self.add_to_queue(self.value_queue, (node, selection))

  def value_thread(self):
    for evaluations in self.queued_items(self.value_queue):
      positions = [node.position for (node, _) in evaluations]
      values = self.values(positions)

      for (node, selection), value in zip(evaluations, values):
        node.set_evaluation(value)
        node.backup_evaluation_result(value, selection)

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
  def rollout(self, node, selection):
    if node.terminal:
      node.backup_rollout_result(node.evaluation, selection)
    else:
      self.add_to_queue(self.rollout_queue, (node, selection))

  def rollout_thread(self):
    for rollouts in self.queued_items(self.rollout_queue):
      self.run_rollouts(rollouts)

  def run_rollouts(self, rollouts):
    positions = [
        self.play_counter_moves(node.position) for (node, _) in rollouts
    ]
    while rollouts:
      moves = self.rollout_moves(positions)
      new_positions, new_rollouts = [], []

      for position, move, rollout in zip(positions, moves, rollouts):
        position = self.play_counter_moves(position.move(move))
        if position.gameover():
          node, selection = rollout
          node.backup_rollout_result(position.result, selection)
        else:
          new_positions.append(position)
          new_rollouts.append(rollout)

      positions = new_positions
      rollouts = new_rollouts

  def play_counter_moves(self, position):
    while position.counter_move is not None:
      position = position.move(position.counter_move)
    return position

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
        self.rollout_network.threats: threats
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
      'position', 'config', 'terminal', 'evaluated', 'value', 'parents',
      'children', 'evaluation', 'evaluation_total', 'evaluation_counts',
      'evaluation_count', 'evaluation_value', 'rollout_total', 'rollout_count',
      'rollout_counts', 'rollout_value', 'priors'
  ]

  def __init__(self, position, config):
    self.position = position
    self.config = config
    self.terminal = position.gameover()
    if self.terminal:
      self.evaluated = True
      self.evaluation = position.result
      self.evaluation_value = position.result
      self.rollout_value = position.result
      self.value = position.result * -util.turn_win(self.position.turn)
    else:
      self.evaluated = False
      self.value = 0
      self.rollout_value = 0
      self.evaluation_value = 0
    self.children = []
    self.parents = set()

    self.evaluation_total = 0
    self.evaluation_count = EPSILON  # Avoid divide by 0
    self.evaluation_counts = None
    self.rollout_total = 0
    self.rollout_count = EPSILON  # Avoid divide by 0
    self.rollout_counts = None

  def expand(self, children):
    num_children = len(children)

    # Default priors proportional to the number of fours the move belongs to
    if num_children > 1:
      ratios = (DISK_FOUR_COUNTS * [self.position.legal_moves]).sum(
          axis=(0, 1))
      child_ratios = ratios[ratios > 0]
      self.priors = child_ratios / child_ratios.sum()
    else:
      self.priors = np.array([1])

    # Set these to EPSILON to avoid divide by 0
    self.rollout_counts = np.tile(EPSILON, num_children)
    self.evaluation_counts = np.tile(EPSILON, num_children)

    self.children = children
    for child in children:
      child.add_parent(self)

  def add_parent(self, parent):
    self.parents |= {parent}

  def all_ancestors(self):
    ancestors = self.parents
    while ancestors:
      new_ancestors = set()
      for ancestor in ancestors:
        yield ancestor
        new_ancestors |= ancestor.parents
      ancestors = new_ancestors

  def update_priors(self, priors):
    if len(priors) != len(self.children):
      print('priors', priors, self.children)
    self.priors = priors

  def backup_virtual_loss(self, selection):
    self.backup_rollout_change(
        value_change=-self.config.virtual_loss,
        count_change=self.config.virtual_loss,
        selection=selection)

  def backup_rollout_result(self, result, selection):
    self.backup_rollout_change(
        value_change=result + self.config.virtual_loss,
        count_change=1 - self.config.virtual_loss,
        selection=selection)

  def backup_rollout_change(self, value_change, count_change, selection):
    self.rollout_total += value_change
    self.rollout_count += count_change
    self.update_rollout_value()

    for selected_node, child_index in selection:
      selected_node.rollout_counts[child_index] += count_change

    for node in self.all_ancestors():
      node.update_rollout_value()

  def update_rollout_value(self):
    if self.rollout_counts is not None:
      total_rollouts = self.rollout_count + self.rollout_counts.sum()
      node_value = self.rollout_total / total_rollouts
      child_proportions = self.rollout_counts / total_rollouts
      child_values = [child.rollout_value for child in self.children]
      self.rollout_value = node_value + (
          child_proportions * child_values).sum()
    else:
      self.rollout_value = self.rollout_total / self.rollout_count

    self.update_combined_value()

  def set_evaluation(self, value):
    self.evaluation = value
    self.evaluated = True

  def backup_evaluation_result(self, value, selection):
    self.evaluation_total += value
    self.evaluation_count += 1
    self.update_evaluation_value()

    for selected_node, child_index in selection:
      selected_node.evaluation_counts[child_index] += 1

    for node in self.all_ancestors():
      node.update_evaluation_value()

  def update_evaluation_value(self):
    if self.evaluation_counts is not None:
      total_evaluations = self.evaluation_count + self.evaluation_counts.sum()
      child_proportions = self.evaluation_counts / total_evaluations
      child_values = [child.evaluation_value for child in self.children]
      node_value = self.evaluation_total / total_evaluations
      self.evaluation_value = (
          (child_proportions * child_values).sum() + node_value)
    else:
      self.evaluation_value = self.evaluation_total / self.evaluation_count

    self.update_combined_value()

  def update_combined_value(self):
    if self.terminal: return

    value = (self.config.rollout_proportion * self.rollout_value +
             (1 - self.config.rollout_proportion) * self.evaluation_value)

    self.value = value * -util.turn_win(self.position.turn)

  def max_value_child(self):
    values = np.array([child.value for child in self.children])
    score = values + self.config.exploration_rate * self.exploration_bonus()
    # Add noise to fairly break ties caused by uniform priors
    score += np.random.uniform(low=0, high=0.0001, size=len(self.children))
    index = np.argmax(score)
    if index > len(self.children):
      print(self.position)
      print('index', index)
      print('children', self.children)
      print('score', score)
      print('priors', self.priors)
      print('rollouts', self.rollout_counts)
      print('exploration_bonus', self.exploration_bonus())
    return index, self.children[index]

  def exploration_bonus(self):
    return (
        self.priors * np.sqrt(self.rollout_counts.sum()) / self.rollout_counts)


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
  from position import Position
  alpha4 = Alpha4(flags.FLAGS)
  position = Position("""
    .......
    ...r...
    ...ry..
    ...yr..
    ...ry..
    .y.ry.r
  """)
  result = alpha4.best_move(position, timeout=3)
  print(result)
