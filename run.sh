#!/bin/bash

# need to check this: https://github.com/fommil/netlib-java - ubuntu instructions

java -Dcom.github.fommil.netlib.BLAS=com.github.fommil.netlib.NativeRefBLAS -Xmx8g -server -jar /home/alex/repos/sml/target/scala-2.12/scala-machine-learning-assembly-0.1.jar | tee run.log
