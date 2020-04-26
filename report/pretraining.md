# Transfer Learning
With the success of many deep learning models, from 29/38 teams competing in the 2017 ImageNet Competition <sup>[1](#imageNetFootNote)</sup> achieving an accuracy of greater than 95% [1][2] to beating the human world champion in Go [3]. Deep learning models are built and trained daily by leading teams from Google and other companies, creating such accurate models requires extensive research, skills and computing power.

_Transfer learning_ is the use of applying one neural network to a new problem, the hope is that the learned architecture can be applied to a new problem. This technique has been proven efficient, especially in image classification where pretrained ImageNet models are transferred and fine-tuned to new image datasets achieve high accuracy [4]. The main benefits to transfer learning are seen when there is a _lack of data_ or when there is a _lack of computing power_.

In these two situations the developer may look into transfer learning to find predefined weights which can be fine-tuned to be used on a new image dataset. This will reduce overall training time and also reduce the need for a large dataset. The reason this works can be seen in [5], the early layers in a trained neural network identify lines, edges, curves. It is not until deeper in the network when objects from the training set can be recognised. Clearly trained networks can be used on other image datasets, as in all contexts the first few layers will be similar, from this point we can fine-tune the deeper layers to fit the new context.  

From a computing power perspective, it requires less time to train the final few layers and fine-tune earlier layers, from a dataset size perspective, less data is required as the earlier layers have mostly already been done for you avoiding overfitting these layers to your dataset.

## Use in Autoencoders
To use this technique in an autoencoder, a pretrained network can be used as the encoder. Such networks are state-of-the-art in object detection and so the bottleneck in the encoder will have a good representation of the input data.

A decoder architecture can be defined which takes this bottleneck back to input size. To train the autoencoder first the decoder will be trained while the pretrained encoder will be untrainable. This allows the decoder to learn the reverse of the encoder.

Once learning has reached a plateau then a discriminative layer as described in [6], this allows us to retain the information found in the pretrained network (especially in the early layers) but also slightly tweak the weights and biases to fit the autoencoder. To do this, lower learning rates are applied to the early layers in the autoencoder which increase through the network. The decoder keeps a constant learning rate which is higher than the encoder so it can be tweaked quicker to fit the encoder.

# References

[1] Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution). _ImageNet Large Scale Visual Recognition Challenge_. IJCV. 2015.

[2] Dave Gershgorn. _The Quartz guide to artificial intelligence: What is it, why is it important, and should we be afraid?_. Quartz, 2017. Retrieved April 2020.

[3] David Silver1*, Aja Huang1*, Chris J. Maddison1, Arthur Guez1, Laurent Sifre1, George van den Driessche1, Julian Schrittwieser1, Ioannis Antonoglou1, Veda Panneershelvam1, Marc Lanctot1, Sander Dieleman1, Dominik Grewe1, John Nham2, Nal Kalchbrenner1, Ilya Sutskever2, Timothy Lillicrap1, Madeleine Leach1, Koray Kavukcuoglu1, Thore Graepel1 and Demis Hassabis1. (* = equal contribution). _Mastering the game of Go with deep neural networks and tree search_. Nature, 2016. __529__(7587), pp. 484–489.

[4] Simon Kornblith, Jonathon Shlens and Quoc V. Le. _Do Better ImageNet Models Transfer Better?_. arXiv, 2019. arXiv:1805.08974

[5] Olah, et al.,. _Feature Visualization_. Distill, 2017.

[6] Jeremy Howard, Sebastian Ruder. _Universal Language Model Fine-tuning for Text Classification_. arXiv, 2018. arXiv:1801.06146.


<a name="imageNetFootNote">1</a>: The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) is an object detection and image classification competition which has been known to push computer vision research but also allow for comparison of results.
