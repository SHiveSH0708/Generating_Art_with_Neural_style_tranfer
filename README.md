# Generating_Art_with_Neural_style_tranfer

Execution

In this project, I have deep learning to compose images in the style of another image (ever wish you could paint like Picasso or Van Gogh?). This is known as *neural style transfer*! This is a technique outlined in [Leon A. Gatys' paper, A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), which is a great read, and you should definitely check it out. 

But, what is neural style transfer?

Neural style transfer is an optimization technique used to take three images, a *content* image, a *style reference* image (such as an artwork by a famous painter), and the *input* image you want to style -- and blend them together such that the input image is transformed to look like the content image, but “painted” in the style of the style image.


For example, let’s take an image of this turtle and Katsushika Hokusai's The Great Wave off Kanagawa:

<img src="https://github.com/tensorflow/models/blob/master/research/nst_blogpost/Green_Sea_Turtle_grazing_seagrass.jpg?raw=1" alt="Drawing" style="width: 200px;"/>
<img src="https://github.com/tensorflow/models/blob/master/research/nst_blogpost/The_Great_Wave_off_Kanagawa.jpg?raw=1" alt="Drawing" style="width: 200px;"/>

[Image of Green Sea Turtle]

Now how would it look like if Hokusai decided to paint the picture of this Turtle exclusively with this style? Something like this?

<img src="https://github.com/tensorflow/models/blob/master/research/nst_blogpost/wave_turtle.png?raw=1" alt="Drawing" style="width: 500px;"/>

Is this magic or just deep learning? Fortunately, this doesn’t involve any witchcraft: style transfer is a fun and interesting technique that showcases the capabilities and internal representations of neural networks.  

The principle of neural style transfer is to define two distance functions, one that describes how different the content of two images are , $L_{content}$, and one that describes the difference between two images in terms of their style, $L_{style}$. Then, given three images, a desired style image, a desired content image, and the input image (initialized with the content image), we try to transform the input image to minimize the content distance with the content image and its style distance with the style image. 
In summary, we’ll take the base input image, a content image that we want to match, and the style image that we want to match. We’ll transform the base input image by minimizing the content and style distances (losses) with backpropagation, creating an image that matches the content of the content image and the style of the style image. 

### Specific concepts that are covered:
In the process, I built practical experience and develop intuition around the following concepts

* *Eager Execution* - use TensorFlow's imperative programming environment that evaluates operations immediately 
* * Using [Functional API](https://keras.io/getting-started/functional-api-guide/) to define a model* - we'll build a subset of our model that will give us access to the necessary intermediate activations using the Functional API 
* *Leveraging feature maps of a pretrained model* - Learn how to use pretrained models and their feature maps 
* *Create custom training loops* - we'll examine how to set up an optimizer to minimize a given loss with respect to input parameters

### The general steps to perform style transfer:

1. Visualize data
2. Basic Preprocessing/preparing our data
3. Set up loss functions 
4. Create model
5. Optimize for loss function

### Key Takeaways
* We built several different loss functions and used backpropagation to transform our input image in order to minimize these losses
** In order to do this we had to load in a pretrained model and use its learned feature maps to describe the content and style representation of our images.
** Our main loss functions were primarily computing the distance in terms of these different representations
* We implemented this with a custom model and eager execution
** We built our custom model with the Functional API
** Eager execution allows us to dynamically work with tensors, using a natural python control flow
** We manipulated tensors directly, which makes debugging and working with tensors easier.
* We iteratively updated our image by applying our optimizers update rules using tf.gradient. The optimizer minimized a given loss with respect to our input image.

---

#### Repository by: Shivesh Gupta
