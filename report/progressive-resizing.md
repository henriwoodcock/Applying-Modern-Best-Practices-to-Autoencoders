# Progressive Resizing
_Progressive resizing_ is a technique which has proven effective for GANS [1], Image Super-Resolution [2] and more recently image classification [3] in which the author used progressive resizing as part of their processing to decrease the time taken to train ImageNet to 18 minutes.

Progressive resizing works by first training the network on smaller lower resolution images and slowly increasing the image resolution throughout training. This intuition is to be able to detect objects in images you do not necessarily need high resolution images, this only becomes important for the fewer images where details matter to determine the class.

The main benefit of progressive resizing is the ability to train quicker (and cheaper) for the first few epochs due to the image resolution being smaller, this means a lot less computing power is required to find those first initial relationships.

## Use in autoencoders
To my knowledge there has not been any results or research done into the use of progressive resizing, and in autoencoders it will work slightly differently.

Progressive resizing works well for many networks like Resnet as it is fairly simple to downsample different sized images, especially through the use of stride in convolutional layers. The reason for this is if a smaller image is put into a network built for a larger network then the smaller resolution image would need less downsampling than the larger. However if we have an image such as:

```python
Image(in, 1, 1)
```

applying a convolutional layer with stride 2 will give us:

```python
Image(out, 1, 1)
```
and so the network is still able to work with this new size.

In autoencoders this is slightly different as we attempt to reverse the downsampling step. Due to different images requiring different amounts of downsampling the same applies to upsampling. However using an upsampling technique whether it is _interpolation_ or a _transposed convolution_, we need to specify an upsampling amount. This will always be applied no matter the image size.

Therefore to use this in autoencoders we either need to train two versions of the network. The first is for the lower resolution images, and the second is then for the higher resolution images. For each increase in resolution we just need to insert a new upsampling layer. This should be fine as we can train the network and then just add a head on top. This means we can use our initial training and the weights on the last layer can be learned independently and then fine-tuned with the rest of the network.

[1] Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen. _Progressive Growing of GANs for Improved Quality, Stability, and Variation_. arXiv, 2018. arXiv:1710.10196.

[2] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, Kyoung Mu Lee. _Enhanced Deep Residual Networks for Single Image Super-Resolution_. arXiv, 2017. arXiv:1707.02921.

[3] Jeremy Howard. _Training Imagenet in 3 hours for $25; and CIFAR10 for $0.26_. 2018. https://www.fast.ai/2018/04/30/dawnbench-fastai/.
