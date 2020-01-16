# tsp_art
TSP art or travelling salesman problem art, is an image which can be drawn with one single line. If you draw it by hand you can use one single stroke without lifting your pencil. 

Its an implementation of tsp art with genetics and nearest neighbor algorithms!

Lets start with my pic

![alt text](https://github.com/salar-shdk/tsp_art/blob/master/pics/me_figure0.jpg)

First it takes an image as input and converts it to black & with image; then makes it pure black and with using a threshold, result is something like this

![alt text](https://github.com/salar-shdk/tsp_art/blob/master/pics/me_figure1.png)

Then extracts some nodes from this pic ( reduces image quality using QUALITY parameter and then every black pixel is a node, and add some irregularity to it), this is the result

![alt text](https://github.com/salar-shdk/tsp_art/blob/master/pics/me_figure2.png)

Now we have the graph, all we need is solve tsp on this graph, there is two method implemented, genetics and nearest neighbor, genetics isn't so good on large graphs and nearest neighbor works better; here is the result

![alt text](https://github.com/salar-shdk/tsp_art/blob/master/pics/me_figure3.png)


# Usage
```
python nearest_neighbor.py img.jpg
```

# Requirements
opencv

numpy

matplotlib
