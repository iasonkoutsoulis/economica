# Author S. Iason Koutsoulis, March 2023.
# This code aims at generating a representative economic agent who will live for T=70 and pass, 
# while endowing his offspring with his knowledge, which is twofold: 
# 1) The state of the world in his last t, X_t and
# 2) his action set, Y_t.
# 
# To achieve this, it seems reasonable to start by (1) understanding what a difference in the state of the world is/can be.
# For instance, it makes sense to Markov-Chain it at first, but Smooth Transitions are also very plausible at a later stage.
# (2) We'll have to (markov/smooth) allow for the change the behavior of the agent depending on the state of the world he is 
# living in. Because training a network revolves around matching X and Y while minimizing loss, our X will be state variables,
# Y will be choice variables, and loss will, for the network, be the negative likelihood of matching its response to X to 
# the theoretical agent's response. 