# economica
Authored by S. Iason Koutsoulis, 2023.

This project aims at building neural networks that mimic the response of the general public to changes in macroeconomic events.

Some ideas are floating around, about which should be the overarching structure of said networks; one could attempt to do something along the lines of 
recurrent, convolutional, autoencoder networks, and also introducing state fluctuations via smooth transitioning or Hensen abrupt changes. 

To achieve these goal, it seems reasonable to start by (1) understanding what a difference in the state of the world is/can be.
For instance, it makes sense to Markov-Chain it at first, but Smooth Transitions are also very plausible at a later stage.
(2) We'll have to (markov/smooth) allow for the change the behavior of the agent depending on the state of the world he is 
living in. Because training a network revolves around matching X and Y while minimizing loss, our X will be state variables,
Y will be choice variables, and loss will, for the network, be the negative likelihood of matching its response to X to 
the theoretical agent's response. 

This code is publicly available for all to play with, and with it, I hope to make meaningful contributions to the science of economics and our society.
