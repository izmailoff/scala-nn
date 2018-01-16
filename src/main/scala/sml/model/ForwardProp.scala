package sml.model

import breeze.linalg.*
import breeze.numerics.{relu, sigmoid}
import sml.{Mat, Vec}

object ForwardProp {

  def forwardPropagation(input: Mat, params: List[(Mat, Vec)]): (Mat, List[(Mat, Mat, Vec, Mat)]) = {
    val (a, caches) = params.init.foldLeft((input, List[(Mat, Mat, Vec, Mat)]())) {
      case ((aPrev, caches), (w, b)) =>
        val (a, cache) = reluActivationForward(aPrev, w, b)
        a -> (caches ::: cache :: Nil)
    }
    val (lastW, lastB) = params.last
    val (aLast, cache) = sigmoidActivationForward(a, lastW, lastB)
    aLast -> (caches ::: cache :: Nil)
  }

  def sigmoidActivationForward(prevActivation: Mat, weights: Mat, biases: Vec): (Mat, (Mat, Mat, Vec, Mat)) = {
    val Z = linearForward(prevActivation, weights, biases)
    val A = sigmoid(Z)
    val linear_and_activation_cache = (prevActivation, weights, biases, Z)
    (A, linear_and_activation_cache)
  }

  def linearForward(prevActivation: Mat, weights: Mat, biases: Vec): Mat = {
    val tmp = weights * prevActivation
    val res = tmp(::, *) + biases
    res
  }

  def reluActivationForward(prevActivation: Mat, weights: Mat, biases: Vec): (Mat, (Mat, Mat, Vec, Mat)) = {
    val Z = linearForward(prevActivation, weights, biases)
    val A = relu(Z)
    val linear_and_activation_cache = (prevActivation, weights, biases, Z)
    (A, linear_and_activation_cache)
  }

  // TODO: consider implementing UFunc or figuring out required type
  //  /*activation: Mat => Mat*/
  //  def linearActivationForward(prevActivation: Mat, weights: Mat, biases: Vec, activation: UFunc): (Mat, Mat) = {
  //    val Z = linearForward(prevActivation, weights, biases)
  //    val A = activation(Z)
  //    (A, Z)
  //  }
  //
  //  def sigmoidActivationForward(prevActivation: Mat, weights: Mat, biases: Vec) =
  //    linearActivationForward(prevActivation, weights, biases, sigmoid)
  //
  //  def reluActivationForward(prevActivation: Mat, weights: Mat, biases: Vec) =
  //    linearActivationForward(prevActivation, weights, biases, relu)
}
