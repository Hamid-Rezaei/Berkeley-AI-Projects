# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        if successorGameState.isWin():
            return 99999
        if successorGameState.isLose():
            return -99999

        newGhostStates = successorGameState.getGhostStates()

        score = successorGameState.getScore()

        currentDistanceToGhosts = self.distanceToGhosts(currentGameState)
        # Here newPos and newGhostStates was used.
        newDistanceToGhosts = self.distanceToGhosts(successorGameState)

        currentDistanceToFoods = self.distanceToFoods(currentGameState)
        # Here newFood, newPos, and newGhostStates was used.
        newDistanceToFoods = self.distanceToFoods(successorGameState)

        currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        currentDistanceToCapsuls = self.distanceToCapsules(currentGameState)
        newDistanceToCapsuls = self.distanceToCapsules(successorGameState)

        # punishment for staying
        if currentGameState.getPacmanPosition() == successorGameState.getPacmanPosition():
            score -= 50

        if min(newDistanceToFoods) < min(currentDistanceToFoods):
            score += 100

        if len(successorGameState.getFood().asList()) < len(currentGameState.getFood().asList()):
            score += 200

        if sum(currentScaredTimes) > 0:
            if min(newDistanceToGhosts) < min(currentDistanceToGhosts):
                score += 300
            if sum(newScaredTimes) == 0:
                score += 300
        else:
            if min(newDistanceToGhosts) > min(currentDistanceToGhosts):
                score += 100

        if newDistanceToCapsuls and (min(newDistanceToCapsuls) < min(currentDistanceToCapsuls)):
            score += 105

        return score

    def distanceToFoods(self, gameState):
        pacmanPos = gameState.getPacmanPosition()
        return [util.manhattanDistance(pacmanPos, foodPos) for foodPos in gameState.getFood().asList()]

    def distanceToGhosts(self, gameState):
        pacmanPos = gameState.getPacmanPosition()
        return [util.manhattanDistance(pacmanPos, ghostPos) for ghostPos in gameState.getGhostPositions()]

    def distanceToCapsules(self, gameState):
        pacmanPos = gameState.getPacmanPosition()
        return [util.manhattanDistance(pacmanPos, capsulePos) for capsulePos in gameState.getCapsules()]


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        action, _ = self.minmax_value(gameState, 0, 0)
        return action

    def minmax_value(self, gameState, index, depth):
        """
        Returns legal actions associated with their utilities ([action, utilities])

        Minimax Implementation (Dispatch)
            if the state is a terminal state: return the state’s utility
            if the next agent is MAX: return max-value(state)
            if the next agent is MIN: return min-value(state)
        """

        # update pacman index
        if index == gameState.getNumAgents():
            index = 0
            depth += 1

        # A terminal state
        if depth == self.depth or len(gameState.getLegalActions(index)) == 0:
            return "Initial action", self.evaluationFunction(gameState)

        # The next agent is MAX -> Pacman with index 0
        if index == 0:
            return self.max_value(gameState, index, depth)
        # The next agent is MIN -> Ghost with index >= 1
        else:
            return self.min_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):
        value = float("-inf")
        best_action = None

        actions = gameState.getLegalActions(index)
        successors = [gameState.generateSuccessor(index, action) for action in actions]
        values = [self.minmax_value(successor, index + 1, depth)[1] for successor in successors]

        for action, v in zip(actions, values):

            if v > value:
                value = v
                best_action = action

        return best_action, value

    def min_value(self, gameState, index, depth):
        value = float("+inf")
        best_action = None

        actions = gameState.getLegalActions(index)
        successors = [gameState.generateSuccessor(index, action) for action in actions]
        values = [self.minmax_value(successor, index + 1, depth)[1] for successor in successors]

        for action, v in zip(actions, values):
            if v < value:
                value = v
                best_action = action

        return best_action, value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        inf = float("inf")
        return self.minmax_value(gameState, 0, 0, -inf, +inf)[0]

    def minmax_value(self, gameState, index, depth, alpha, beta):

        # update pacman index
        if index == gameState.getNumAgents():
            index = 0
            depth += 1

        # A terminal state
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return "Initial action", self.evaluationFunction(gameState)

        # The next agent is MAX -> Pacman with index 0
        if index == 0:
            return self.max_value(gameState, index, depth, alpha, beta)
        # The next agent is MIN -> Ghost with index >= 1
        else:
            return self.min_value(gameState, index, depth, alpha, beta)

    def max_value(self, gameState, index, depth, alpha, beta):
        value = float("-inf")
        best_action = None

        actions = gameState.getLegalActions(index)
        for action in actions:
            successor = gameState.generateSuccessor(index, action)

            _, v = self.minmax_value(successor, index + 1, depth, alpha, beta)
            if v > value:
                value = v
                best_action = action

            if v > beta:
                return action, v

            alpha = max(alpha, v)

        return best_action, value

    def min_value(self, gameState, index, depth, alpha, beta):
        value = float("+inf")
        best_action = None

        actions = gameState.getLegalActions(index)
        for action in actions:
            successor = gameState.generateSuccessor(index, action)

            _, v = self.minmax_value(successor, index + 1, depth, alpha, beta)
            if v < value:
                value = v
                best_action = action

            if v < alpha:
                return action, v

            beta = min(beta, v)

        return best_action, value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, _ = self.expectimax(gameState, 0, 0)
        return action

    def expectimax(self, gameState, index, depth):

        # update pacman index (it's like a loop)
        if index == gameState.getNumAgents():
            index = 0
            depth += 1

        # A terminal state
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return "Initial action", self.evaluationFunction(gameState)

        # The next agent is MAX -> Pacman with index 0
        if index == 0:
            return self.max_value(gameState, index, depth)

        # The next agent is Expected -> Ghost with index >= 1
        else:
            return self.expected_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):
        value = float("-inf")
        best_action = "random action"

        actions = gameState.getLegalActions(index)
        for action in actions:
            successor = gameState.generateSuccessor(index, action)

            _, v = self.expectimax(successor, index + 1, depth)

            if v > value:
                value = v
                best_action = action

        return best_action, value

    def expected_value(self, gameState, index, depth):

        actions = gameState.getLegalActions(index)
        p = 1.0 / len(actions)
        successors = [gameState.generateSuccessor(index, action) for action in actions]
        values = [p * self.expectimax(successor, index + 1, depth)[1] for successor in successors]

        return "", sum(values)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()
    
    "*** YOUR CODE HERE ***"
    ghostsDistance = [util.manhattanDistance(pacmanPosition, ghost) for ghost in ghostPositions]
    closestGhost = 0 if len(ghostsDistance) == 0 else min(ghostsDistance)

    foodsDistance = [util.manhattanDistance(pacmanPosition, food) for food in foods.asList()]
    closestFood = 1 if len(foodsDistance) == 0 else min(foodsDistance)

    capsuleDistance = [util.manhattanDistance(pacmanPosition, capsule) for capsule in currentGameState.getCapsules()]
    closetCapsule = 1 if len(capsuleDistance) == 0 else min(capsuleDistance)

    periodOfScared = sum(scaredTimers)

    return (0.4 * currentGameState.getScore() +
            0.2 * (-2 * periodOfScared + 1) * closestGhost +
            0.5 * 1 / closestFood +
            10.0 / closetCapsule
            )


# Abbreviation
better = betterEvaluationFunction
