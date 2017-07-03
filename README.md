# Alpha4

Alpha4 plays [Connect4](https://en.wikipedia.org/wiki/Connect_Four) in the style of [AlphaGo](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf). Specifically, it consists of policy and value neural networks embedded in Monte-Carlo Tree Search. The policy network is trained by reinforcement learning against some simple opponents and old versions of itself. Once the policy network is trained it is used to generate a large collection of positions which are used to train the value network.

Alpha4 is pretty strong. Connect4 is a solved game and it is known that the first player always wins. When Alpha4 plays first, it occasionally plays the perfect game to bring home a beautiful victory.

## Requirements
- Python - tested with v3.5
- Tensorflow - tested with v1.1.0
- Tkinter - normally included with python

## Play it
- Play the pretrained networks `python gui.py`
- Show winning moves while playing `python gui.py --threats`
- If you can't beat it then you could always [cheat](http://connect4.gamesolver.org/)!

## Policy Training
Train a policy network with `python policy_training.py`

The policy network is trained against increasingly strong sets of opponents. To start with it plays against some simple handcoded algorithms
- `RandomPlayer` plays randomly with a bias towards central disks
- `RandomThreatPlayer` makes most moves like `RandomPlayer` but always plays winning moves or blocks the opponent's winning moves
- `MaxThreatPlayer` plays like `RandomThreatPlayer` except it also tries to create winning moves

Once the policy network can consistently beat them a clone of the current policy is created and added as an additional opponent. More and more clones are added as opponents as the network becomes stronger. Checkout tensorboard to see win rates against the various opponents created - `tensorboard --logdir runs`

`REINFORCE` is used to train the network with the result (+1 for a win, 0 for a draw, -1 for a loss) used as the reward for each move played during a game.

Many games of Connect4 can be very similar so the games are initialised with some random moves to give the network a more diverse set of positions to train on. Entropy regularisation is also used to ensure the network doesn't end up with a single fixed policy.

## Value Training
Once the policy network is trained it can be used to generate a large quantity of positions

`python generate_rollout_positions.py`

These can then be used to supervise the training of a value network. The positions are split into training and validation sets. At the end of each epoch the validation score of the network is calculated and the network is only saved if the validation score improves

`python value_training.py`

## Monte-Carlo Tree Search details
- Multithreaded 
    - MCTS threads add nodes to a prior queue, a rollout queue and a value queue for batch processing
    - Prior threads use a policy network to calculate the move priors for each position
    - Rollout threads use another policy network to rollout positions until the end of the game
    - Value threads use a value network to estimate the value of the position
- All updates from the above threads are *optimistically* applied [lock-free](https://webdocs.cs.ualberta.ca/~mmueller/ps/enzenberger-mueller-acg12.pdf)
- Because Connect4 has a low branching factor many combinations of moves lead to the same position, known as *transpositions*. Rollouts and values estimates are applied to all branches of the search tree that lead to the end node. This is called UCT3 and is fully described [here](http://alum.wpi.edu/~jbrodeur/cig08.pdf)
- Despite the policy networks being trained with entropy regularisation they don't have enough entropy in the priors or diversity in the rollouts. This is remedied by appying high temperatures to the final softmax layers in these networks
- The performance is limited by the throughput of the policy and value networks. By using threads we at least ensure that the networks are busy most of the time and are processing reasonable size batches

## Credits
- [AlphaGo](https://deepmind.com/research/alphago/)
- [A Lock-free Multithreaded Monte-Carlo Tree Search
Algorithm](https://webdocs.cs.ualberta.ca/~mmueller/ps/enzenberger-mueller-acg12.pdf)
- [Transpositions and Move Groups in Monte Carlo Tree Search](http://alum.wpi.edu/~jbrodeur/cig08.pdf)
- The GUI was inspired by [github.com/mevdschee/python-connect4](https://github.com/mevdschee/python-connect4/blob/master/connect4.py)
