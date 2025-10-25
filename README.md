A supervised, convolutional neural network to recognise images from the MNIST dataset. I have used packages minimally to acquire a thorough understanding of machine learning
from its mathematical  foundations.

Packages used: 
- matplotlib - for a tracking accuracy and cost, and for visually representing training images during debugging
- NumPy - for linear algebra operations

Features:
- Choice between quadratic and cross entropy cost functions
- L2 regularisation to reduce dependency on a few neurons (can be turned off with REGULARISATION_PARAMETER = 0)
- Early stopping strategy to reduce overfitting and minimise training time
- Choice between an improved weight initialisation (reducing the likelihood of neuron saturation) and the original strategy
- Detailed tracking of cost and accuracy, including unregularised cost, regularisation cost, total cost, and both training and testing accuracy.

The inclusion of these features (in the second commit) have improved accuracy from around 95% to 98%.

This project was part of my personal exploration of deep learning fundamentals, focusing on understanding neural networks without relying on high-level frameworks.

I was really proud of making and improving this project. I have noticed my strength in vector calculus improve, and I've grown in familiarity with NumPy  - I have really noticed the performance it is capable of when dealing with linear algebra in Python! I have been delighted by the instances where the underlying maths and notation 'clicks', and when that understanding translates into working code. Each iteration of adding a new feature, fixing bugs, adjusting hyperparameters and seeing the improvements in results was immensely rewarding, and fueled me to add 'just one more' change to eke out performance. 

My biggest takeaway from this project is the realisation of how big a problem overfitting can be. For example, each time I added a new feature or adjusted a hyperparameter
I questioned whether the improvement would generalise to real-world data. Even with techniques such as cross validation, regularisation, validation datasets and early stopping, overfitting is near impossible to eliminate completely. One particular realisation was that tuning hyperparameters may just overfit the data to the validation set. Changing the architecture and re-tuning could then overfit data to the test set, and so on. I would love to explore this concept and potential solutions in the future! More generally, this taught me to critically evaluate models and results. An increase in validation or test accuracy doesn't necessarily correlate to an improvement in real-world results - you must always test the model for robustness and ensure a more complex model is delivering better results. 

Contact info:
fg550@bath.ac.uk
