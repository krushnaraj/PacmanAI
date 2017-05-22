# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
#
import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
    #print self.values['test']
    #self.values['test'] +=1
    for x in range(0,iterations):
        statevalues = util.Counter()
        for state in mdp.getStates():
            if mdp.isTerminal(state):
                statevalues[state]=0
                continue
            actionvalue = -100000
            for action in mdp.getPossibleActions(state):
                S_next = self.getQValue(state,action)
                if actionvalue<S_next:
                    actionvalue = S_next
            statevalues[state] = actionvalue
        self.values = statevalues

    "*** YOUR CODE HERE ***"
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    S_next_val = 0
    for nextstate in self.mdp.getTransitionStatesAndProbs(state, action):
        R = self.mdp.getReward(state, action, nextstate[0])
        S_next_val += nextstate[1] * (R + (self.discount * self.values[nextstate[0]]))

    return S_next_val
    #util.raiseNotDefined()

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    bestActionValue = -100000
    bestAction = None
    nextactions = self.mdp.getPossibleActions(state)

    if self.mdp.isTerminal(state):
        return None

    for action in nextactions:
        S_next_vals = self.getQValue(state, action)
        if bestActionValue < S_next_vals:
            bestAction = action
            bestActionValue = S_next_vals
    return bestAction
    #util.raiseNotDefined()

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    bestActionValue = -100000
    bestAction = None
    nextactions = self.mdp.getPossibleActions(state)

    if self.mdp.isTerminal(state):
        return None

    for action in nextactions:
        S_next_vals = self.getQValue(state, action)
        if bestActionValue < S_next_vals:
            bestAction = action
            bestActionValue = S_next_vals
    return bestAction
    #return self.getPolicy(state)
  
