from datetime import datetime
import os
import tensorflow as tf
import threading
import time


def run_directory(config):
  def find_previous_run(dir):
    if os.path.isdir(dir):
      runs = [child[4:] for child in os.listdir(dir) if child[:4] == 'run_']
      if runs:
        return max([int(run) for run in runs])

    return 0

  if config.run_dir == 'latest':
    parent_dir = 'runs/'
    previous_run = find_previous_run(parent_dir)
    run_dir = parent_dir + ('run_%d' % previous_run)
  elif config.run_dir:
    run_dir = config.run_dir
  else:
    parent_dir = 'runs/'
    previous_run = find_previous_run(parent_dir)
    run_dir = parent_dir + ('run_%d' % (previous_run + 1))

  if run_dir[-1] != '/':
    run_dir += '/'

  if not os.path.isdir(run_dir):
    os.makedirs(run_dir)

  print('Checkpoint and summary directory is %s' % run_dir)

  return run_dir


def turn_win(turn):
  return turn * -2 + 1  # RED = +1, YELLOW = -1


def restore(session, run_dir, network):
  latest_checkpoint = tf.train.latest_checkpoint(run_dir,
                                                 network.scope + '_checkpoint')
  if latest_checkpoint:
    tf.train.Saver(network.variables).restore(session, latest_checkpoint)
    print('Restoring checkpoint %s' % latest_checkpoint)
    return True
  else:
    return False


def try_restore(session, run_dir, network):
  if not restore(session, run_dir, network):
    raise Exception('Checkpoint %s not found in %s' % (network.scope, run_dir))


def save(session, run_dir, network):
  os.makedirs(run_dir, exist_ok=True)
  tf.train.Saver(network.variables).save(
      session,
      os.path.join(run_dir, network.scope + '.ckpt'),
      latest_filename=network.scope + '_checkpoint')
