# Feature Loss
Feature loss is a type of loss function used mainly in style transfer and super resolution networks. First defined in [1] as _perceptual loss_, the loss function captures features instead of pixel by pixel loss between the prediction and ground truth images.

_Perceptual loss_ works by using pretrained image classification models and extracting the features in the ground truth vs the prediction. [2] shows us how different layers in convolutional networks extract different features, from shapes to patterns to human faces.

To check how accurately our network is creating these features we can compare the activations in layers of pretrained networks and minimise the difference between these. This will allow us to create a network which focus on creating the features in them and not just a pixel per pixel loss.

## Use in Autoencoders
Although not seen specifically used in autoencoders, the use of featureloss has been seen in U-Net architectures (insert reference) for super resolution and style transfer as mentioned before.

One issue found when using an autoencoder for images is that as the images get larger it becomes harder to extract a lot of the information into a smaller bottleneck. The resulting output from the autoencoder is often a blurred image. However when using feature loss the hope is that even if some blur exists we are able to see the features that exist on the image. In some situations this may be more helpful than a full image recreation.

# References
[1] Justin Johnson, Alexandre Alahi and Li Fei-Fei. _Perceptual Losses for Real-Time Style Transfer and Super-Resolution_. arXiv, 2016. arXiv:1603.08155

[2] Zeiler, Matthew & Fergus, Rob. _Visualizing and Understanding Convolutional Neural Networks_. ECCV 2014, Part I, LNCS 8689. 8689. 10.1007/978-3-319-10590-1_53.
