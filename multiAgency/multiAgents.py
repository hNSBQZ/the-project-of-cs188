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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        closest = 42069
        newFoodList = newFood.asList()  # a list of (x,y) in newFood
        hasFood = 0  # can not use the isWin to determine if there is any FOOD left, use hasFood boolean to figure that out
        heuristicFood = 0
        for (x, y) in newFoodList:
            if newFood[x][y]:  # if there is a Food
                MD = manhattanDistance(newPos, (x, y))
                hasFood = 1
                if (MD < closest):  # choose the distance to the closest food
                    closest = MD
        if hasFood == 0:
            return successorGameState.getScore()
        else:
            heuristicFood = 1 / (closest + 1)  # heuristic is the reciprocal (inverse) of the distance to closest food
            return heuristicFood + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        agent_nums=gameState.getNumAgents()
        nexts=gameState.getLegalActions(0)
        maxUtility=-float('inf')
        ans=None
        def MiniMax(turn:int,state,depth)->int:
            if state.isWin() or state.isLose() or depth==self.depth:
                return self.evaluationFunction(state)
            if turn==0:
                v=-float('inf')
                for a in state.getLegalActions(turn):
                    v=max(v,MiniMax(1,state.generateSuccessor(turn,a),depth))
            else:
                v=float('inf')
                if turn==agent_nums-1:
                    depth+=1
                for a in state.getLegalActions(turn):
                    v = min(v, MiniMax((turn+1)%agent_nums, state.generateSuccessor(turn,a), depth))
            return v
        for next in nexts:
            nextState=gameState.generateSuccessor(0,next)
            v=MiniMax(1,nextState,0)
            print(v)
            if v>maxUtility:
                maxUtility=v
                ans=next
        return ans
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agent_nums = gameState.getNumAgents()
        nexts = gameState.getLegalActions(0)
        maxUtility = -float('inf')
        ans = None

        def MiniMax(turn: int, state, depth,alpha,beta) -> int:
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if turn == 0:
                v = -float('inf')
                for a in state.getLegalActions(turn):
                    v = max(v, MiniMax(1, state.generateSuccessor(turn, a), depth,alpha,beta))
                    if v>beta:
                        return v
                    alpha=max(alpha,v)
                return v
            else:
                v = float('inf')
                if turn == agent_nums - 1:
                    depth += 1
                for a in state.getLegalActions(turn):
                    v = min(v, MiniMax((turn + 1) % agent_nums, state.generateSuccessor(turn, a), depth,alpha,beta))
                    if v<alpha:
                        return v
                    beta=min(beta,v)
                return v
        alpha_1=-float('inf')
        for next in nexts:
            nextState = gameState.generateSuccessor(0, next)
            v = MiniMax(1, nextState, 0,alpha_1,float('inf'))
            if v > maxUtility:
                maxUtility = v
                ans = next
            alpha_1=max(alpha_1,v)
        return ans
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        agent_nums = gameState.getNumAgents()
        nexts = gameState.getLegalActions(0)
        maxUtility = -float('inf')
        ans = None

        def Expectimax(turn:int,state,depth)->int:
            if state.isWin() or state.isLose() or depth==self.depth:
                return self.evaluationFunction(state)
            if turn==0:
                v=-float('inf')
                for a in state.getLegalActions(turn):
                    v=max(v,Expectimax(1,state.generateSuccessor(turn,a),depth))
            else:
                v=0
                if turn==agent_nums-1:
                    depth+=1
                cnt=len(state.getLegalActions(turn))
                for a in state.getLegalActions(turn):
                    v += Expectimax((turn+1)%agent_nums, state.generateSuccessor(turn,a), depth)
                v=v/cnt
            return v
        for next in nexts:
            nextState=gameState.generateSuccessor(0,next)
            v=Expectimax(1,nextState,0)
            print(v)
            if v>maxUtility:
                maxUtility=v
                ans=next
        return ans

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    walls = currentGameState.getWalls()

    # 如果不是新的ScaredTimes，则新状态为ghost：返回最低值

    newFood = newFood.asList()
    ghostPos = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
    scared = min(newScaredTimes) > 0

    if currentGameState.isLose():
        return float('-inf')

    if newPos in ghostPos:
        return float('-inf')

    # 如果不是新的ScaredTimes，则新状态为ghost：返回最低值

    closestFoodDist = sorted(newFood, key=lambda fDist: util.manhattanDistance(fDist, newPos))
    closestGhostDist = sorted(ghostPos, key=lambda gDist: util.manhattanDistance(gDist, newPos))

    score = 0

    fd = lambda fDis: util.manhattanDistance(fDis, newPos)
    gd = lambda gDis: util.manhattanDistance(gDis, newPos)

    if gd(closestGhostDist[0]) < 3:
        score -= 300
    if gd(closestGhostDist[0]) < 2:
        score -= 1000
    if gd(closestGhostDist[0]) < 1:
        return float('-inf')

    if len(currentGameState.getCapsules()) < 2:
        score += 100

    if len(closestFoodDist) == 0 or len(closestGhostDist) == 0:
        score += scoreEvaluationFunction(currentGameState) + 10
    else:
        score += (scoreEvaluationFunction(currentGameState) + 10 / fd(closestFoodDist[0]) + 1 / gd(
            closestGhostDist[0]) + 1 / gd(closestGhostDist[-1]))

    return score



        # util.raiseNotDefined()

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
