# scala-nn
A Neural Network (NN) implemented in Scala from scratch.

## What?
This is just the initial implementation of a vanilla NN that classifies images of ... cats!

I tried to do everything in a functional way, avoiding mutation and side effects. A cute idea to toy with is that forward prop is `foldLeft` and back prop is `foldRight`.

The initial code layout follows Deep Learning Coursera's course by Andrew Ng code organization into functions (Python). I use the same terminology when naming variables and same network structure with forward prop caches, etc. I also used the data for testing from the course.

This is only the start. There are many improvements that can be made.

## Why?
To learn by implementing and to see design tradeoffs in a typesafe statically typed language - Scala.

## What's Next?
If I find time I'll introduce better abstractions for all configurable algorithms and settings. I also have some ideas how to define NN in a more typesafe way than this or regular frameworks do.

## Quick Start
Have java runtime JVM 8 installed. To build run from shell:

    > ./sbt assembly

This will download dependencies, compile the code, run all the tests and create a self-contained JAR file. After the build is done run the JAR with:

	> java -jar target/scala-2.12/scala-machine-learning-assembly-0.1.jar

Default settings can be changed via `-D` Java arguments. For more info see `reference.conf` file.

## Dataset
Each image is of size `(width_px, height_px, 3)` - 3 channels (RGB).

Number of training examples: 209

Number of test examples: 50

Each image is of size: (64, 64, 3)

    train_x_orig shape: (209, 64, 64, 3)
    train_y shape: (1, 209)
    test_x_orig shape: (50, 64, 64, 3)
    test_y shape: (1, 50) 
    train_x's shape: (12288, 209)
    test_x's shape: (12288, 50)

12,288 equals 64×64×3 which is the size of one reshaped image vector.
