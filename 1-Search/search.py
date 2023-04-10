# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
from typing import Optional

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


# def graphSearch(problem, fringe):
#     closed = set()
#
#     # push to fringe state and list of actions to it.
#     fringe.push((problem.getStartState(), []))
#     to_goal_actions = []
#
#     while not fringe.isEmpty():
#         node, actions = fringe.pop()
#
#         if problem.isGoalState(node):
#             to_goal_actions = actions[:]
#             break
#
#         if not (
#                  node in closed
#         ):
#             closed.add(node)
#
#             for child_node, action, cost in problem.getSuccessors(node):
#                 fringe.push((child_node, actions + [action]))
#
#     return to_goal_actions


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """
    stack = util.Stack()
    return graphSearch(problem=problem, fringe=stack)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()
    return graphSearch(problem=problem, fringe=queue)


def uniformCostSearch(problem):
    """Search the node of the least total cost first."""
    priority_queue = util.PriorityQueue()
    return graphSearch(problem=problem, fringe=priority_queue, init_cost=0)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    priority_queue = util.PriorityQueue()
    return graphSearch(problem=problem, fringe=priority_queue, init_cost=0, heuristic=heuristic)


def iterativeDeepeningSearch(problem):
    for depth in range(1, 100):
        actions = depthLimitedSearch(problem, depth)
        if actions is not None:
            return actions


def depthLimitedSearch(problem, depth):
    fringe = util.Stack()
    closed = set()

    fringe.push((problem.getStartState(), []))
    while not fringe.isEmpty():
        node, actions = fringe.pop()

        if problem.isGoalState(node):
            return actions

        if not node in closed and len(actions) < depth:
            closed.add(node)
            for child_node, action, cost in problem.getSuccessors(node):
                if child_node not in closed:
                    fringe.push((child_node, actions + [action]))

    return None


def graphSearch(
    problem,
    fringe,
    init_cost: Optional[int] = 1,
    heuristic=nullHeuristic
) -> list:

    closed = set()

    # push to fringe state and list of actions to it.
    fringe.push((problem.getStartState(), []), init_cost)
    actions_to_goal = []

    while not fringe.isEmpty():
        node, actions = fringe.pop()

        if problem.isGoalState(node):
            actions_to_goal = actions[:]
            break

        if not (
                 node in closed
        ):
            closed.add(node)

            for child_node, action, cost in problem.getSuccessors(node):
                if child_node not in closed:
                    total_cost = problem.getCostOfActions(actions + [action]) + heuristic(child_node, problem)
                    fringe.push((child_node, actions + [action]), total_cost)

    return actions_to_goal


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
