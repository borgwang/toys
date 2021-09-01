## Monte-Carlo Tree Search for Tic-Tac-Toe
MCTS is a popular planning algorithm which was widely used in Game AI.
Vanilla MCTS randomly picks an action when building search tree. UCT instead uses an algorithm that can trade-off exploration and exploitation. For more details about UCT, refer [MCTS survey](http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf).    

## Tic-Tac-Toe
Here we use UCT1 to play an variation of classic Tic-Tac-Toe called 'Ultimate Tic-Tac-Toe'.
See [this post]( http://mathwithbaddrawings.com/2013/06/16/ultimate-tic-tac-toe/ ) for details about the game.
The implementations of the game borrows from [this repo](https://github.com/jbradberry/ultimate_tictactoe).

## Screenshot
<img src="https://github.com/borgwang/toys/raw/master/algorithm-MCTS/mcts.png" width="300" height="430" alt="mcts" align=center />

## Reference
* [MCTS survey](http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf)
* [Jeff Bradberry's post](http://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/)
