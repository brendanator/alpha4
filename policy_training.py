from consts import *
from network import PolicyNetwork
import numpy as np
import os
from players import *
from position import Position
import tensorflow as tf
import util

flags = tf.app.flags
flags.DEFINE_string("run_dir", None, "Run directory")
flags.DEFINE_integer("batch_size", 256, "Batch size")
flags.DEFINE_integer("batches", 10000, "Number of batches")
flags.DEFINE_float("entropy", 0.03, "Entropy regularisation rate")
flags.DEFINE_float("learning_rate", 0.001, "Adam learning rate")
config = flags.FLAGS


class PolicyTraining(object):
    def __init__(self, config):
        self.config = config
        self.run_dir = util.run_directory(config)

        self.session = tf.Session(
            config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        )

        self.policy_network = PolicyNetwork("policy")
        self.policy_player = PolicyPlayer(self.policy_network, self.session)
        util.restore_or_initialize_network(
            self.session, self.run_dir, self.policy_network
        )

        # Train ops
        self.create_train_op(self.policy_network)
        self.writer = tf.summary.FileWriter(self.run_dir)
        util.restore_or_initialize_scope(
            self.session, self.run_dir, self.training_scope.name
        )

        self.opponents = Opponents(
            [RandomPlayer(), RandomThreatPlayer(), MaxThreatPlayer()]
        )
        self.opponents.restore_networks(self.session, self.run_dir)

    def create_train_op(self, policy_network):
        with tf.variable_scope("policy_training") as self.training_scope:
            self.move = tf.placeholder(tf.int32, shape=[None], name="move")
            self.result = tf.placeholder(tf.float32, shape=[None], name="result")

            policy = tf.reshape(policy_network.policy, [-1, HEIGHT, WIDTH])
            move = tf.expand_dims(tf.one_hot(self.move, WIDTH), axis=1)
            turn = util.turn_win(policy_network.turn)
            move_probability = tf.reduce_sum(policy * move, axis=[1, 2])

            result_loss = -tf.reduce_mean(tf.log(move_probability) * turn * self.result)
            entropy_regularisation = -config.entropy * tf.reduce_mean(
                policy_network.entropy
            )
            loss = result_loss + entropy_regularisation

            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self.train_op = optimizer.minimize(loss, self.global_step)

            # Summary
            tf.summary.scalar("loss", loss)
            for var in policy_network.variables + policy_network.policy_layers:
                tf.summary.histogram(var.name, var)
            self.summary = tf.summary.merge_all()

    def train(self):
        for _ in range(self.config.batches):
            opponent = self.opponents.choose_opponent()
            games = self.play_games(opponent)
            step, summary = self.train_games(opponent, games)
            self.process_results(opponent, games, step, summary)

            if self.opponents.all_beaten():
                name = self.opponents.next_network_name()
                print("All opponents beaten. Creating %s" % name)
                self.create_new_opponent(name)

            if step % 100 == 0:
                self.save()

        self.save()

    def save(self):
        util.save_network(self.session, self.run_dir, self.policy_network)
        util.save_scope(self.session, self.run_dir, self.training_scope.name)
        self.opponents.save_opponent_stats(self.run_dir)

    def play_games(self, opponent):
        # Create games
        games = incomplete_games = [Game() for _ in range(self.config.batch_size)]

        # Let opponent play first in half of the games
        self.play_move(games[0 : len(games) // 2], opponent)
        player = self.policy_player

        while incomplete_games:
            self.play_move(incomplete_games, player)
            player = self.policy_player if player != self.policy_player else opponent
            incomplete_games = [
                game for game in incomplete_games if not game.position.gameover()
            ]

        return games

    def play_move(self, games, player):
        positions = [game.position for game in games]
        moves = player.play(positions)

        for game, move in zip(games, moves):
            game.move(move, player == self.policy_player)

    def train_games(self, opponent, games):
        turn, disks, empty, legal_moves, threats, moves, results = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for game in games:
            for position, move in game.policy_player_moves:
                turn.append(position.turn)
                disks.append(position.disks)
                empty.append(position.empty)
                legal_moves.append(position.legal_moves)
                threats.append(position.threats)
                moves.append(move)
                results.append(game.result)

        _, step, summary = self.session.run(
            [self.train_op, self.global_step, self.summary],
            {
                self.policy_network.turn: turn,
                self.policy_network.disks: disks,
                self.policy_network.empty: empty,
                self.policy_network.legal_moves: legal_moves,
                self.policy_network.threats: threats,
                self.move: moves,
                self.result: results,
            },
        )

        return step, summary

    def process_results(self, opponent, games, step, summary):
        win_rate = np.mean([game.policy_player_score for game in games])
        average_moves = sum(len(game.moves) for game in games) / self.config.batch_size

        opponent_summary = tf.Summary()
        opponent_summary.value.add(
            tag=self.training_scope.name + "/" + opponent.name + "/win_rate",
            simple_value=win_rate,
        )
        opponent_summary.value.add(
            tag=self.training_scope.name + "/" + opponent.name + "/moves",
            simple_value=average_moves,
        )

        self.writer.add_summary(summary, step)
        self.writer.add_summary(opponent_summary, step)

        self.opponents.update_win_rate(opponent, win_rate)

        print(
            "Step %d. Opponent %s, win rate %.2f <%.2f>, %.2f moves"
            % (
                step,
                opponent.name,
                win_rate,
                self.opponents.win_rates[opponent],
                average_moves,
            )
        )

    def create_new_opponent(self, name):
        # Create clone of policy_player
        clone = PolicyNetwork(name)
        self.session.run(self.policy_network.assign(clone))
        util.save_network(self.session, self.run_dir, clone)
        new_opponent = PolicyPlayer(clone, self.session)

        self.opponents.decrease_win_rates()
        self.opponents.add_opponent(new_opponent)


class Opponents(object):
    def __init__(self, opponents):
        self.win_rates = {}
        for opponent in opponents:
            self.add_opponent(opponent)

    def add_opponent(self, opponent):
        self.win_rates[opponent] = EPSILON

    def decrease_win_rates(self):
        # Decrease win rate so tough players must be beaten again
        self.win_rates = {
            opponent: max(2 * win_rate - 1, EPSILON)
            for opponent, win_rate in self.win_rates.items()
        }

    def update_win_rate(self, opponent, win_rate):
        # Win rate is a moving average
        self.win_rates[opponent] = self.win_rates[opponent] * 0.9 + win_rate * 0.1

    def all_beaten(self):
        result = True
        for win_rate in self.win_rates.values():
            result = result and win_rate > 0.7
        return result

    def choose_opponent(self):
        # More difficult opponents are chosen more often
        win_rates = np.maximum(list(self.win_rates.values()), 0.1)
        probs = (1 / win_rates ** 2) - 1
        normalised_probs = probs / probs.sum()
        return np.random.choice(list(self.win_rates.keys()), p=normalised_probs)

    def next_network_name(self):
        network_opponents = len(
            [
                opponent
                for opponent in self.win_rates.keys()
                if type(opponent) == PolicyPlayer
            ]
        )
        return "network-%d" % (network_opponents + 1)

    def save_opponent_stats(self, run_dir):
        with open(os.path.join(run_dir, "opponents"), "w") as f:
            f.write(
                "\n".join(
                    [
                        opponent.name + " " + str(win_rate)
                        for opponent, win_rate in sorted(
                            self.win_rates.items(), key=lambda x: x[1]
                        )
                    ]
                )
            )

    def restore_networks(self, session, run_dir):
        opponents_file = os.path.join(run_dir, "opponents")
        if os.path.exists(opponents_file):
            with open(opponents_file) as f:
                for line in f.readlines():
                    opponent_name, win_rate_string = line.strip().split()
                    win_rate = float(win_rate_string)
                    if opponent_name[:8] == "network-":
                        print("Restoring %s" % opponent_name)
                        network = PolicyNetwork(opponent_name)
                        util.restore_network_or_fail(session, run_dir, network)
                        opponent = PolicyPlayer(network, session)
                        self.win_rates[opponent] = win_rate
                    else:
                        for opponent in self.win_rates.keys():
                            if opponent_name == opponent.name:
                                self.win_rates[opponent] = win_rate


class Game(object):
    def __init__(self):
        self.position = Position()
        self.positions = [self.position]
        self.moves = []
        self.policy_player_moves = []
        self.result = None

        # Make it equally likely to train on red as yellow
        if np.random.rand() < 0.5:
            self.move(np.random.choice(self.position.legal_columns()))

        # Setup a random position
        while np.random.rand() < 0.75:
            self.move(np.random.choice(self.position.legal_columns()))

    def move(self, move, policy_player_turn=False):
        if policy_player_turn:
            self.policy_player_moves.append((self.position, move))
        self.moves.append(move)
        self.position = self.position.move(move)
        self.positions.append(self.position)
        if self.position.gameover():
            self.result = self.position.result
            self.policy_player_score = float(policy_player_turn) if self.result else 0.5


def main(_):
    training = PolicyTraining(config)
    training.train()


if __name__ == "__main__":
    tf.app.run()
