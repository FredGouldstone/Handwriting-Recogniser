A supervised, convolutional neural network to recognise images from the MNIST dataset. I have used packages minimally to acquire a thorough understanding of machine learning
from its mathematical  foundations.

Packages used: 
- matplotlib (tracking accuracy and cost, visually representing the training images for debugging)
- NumPy (linear algebra)

Features:
- Choice between quadratic and cross entropy cost functions
- L2 regularisation (can be turned off with REGULARISATION_PARAMETER = 0)
- No-improvement-in-n-epochs strategy to minimise training time
- Choice between an improved weight initialistion for reduced likelihood of neuron saturation and the original strategy
- Detailed tracking of cost and accuracy (can see unregularised cost, regularisation cost and total cost as well as training and testing cost)

The inclusions of these features (in second commit) have improved accuracy from around 95% to 98%.

I was really proud of making and improving this project. I have become more comfortable working my way through vector calculus and improved my debugging techniques. 
I'm also enjoying the sense of comfortability I have with NumPy - I have really noticed the performance it is capable of when dealing with linear algebra in python!
I really enjoyed making the underlying maths and notation 'click', and then seeing it all integrate into my program. Each iteration of adding a new feature, 
fixing bugs, adjusting hyperparameters and seeing the improvements in results was immensely rewarding, and fueled me to add 'just one more' change to eak out performance. 

Contact info:
fg550@bath.ac.uk
