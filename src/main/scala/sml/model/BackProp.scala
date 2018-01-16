package sml.model

import breeze.linalg.{*, sum}
import breeze.numerics.sigmoid
import sml.{Mat, Vec}

object BackProp {

  def backwardPropagation(AL: Mat, labels: Vec, caches: List[(Mat, Mat, Vec, Mat)]): List[(Mat, Mat, Vec)] = {
    val Y = labels.toDenseMatrix
    val dAlast = -((Y /:/ AL) - ((1.0 - Y) /:/ (1.0 - AL)))
    val (lastA, w, b, lastAcache) = caches.last
    val lastGrad@(dA, dW, db) = linearActivationBackward(dAlast, lastA, w, b, lastAcache, sigmoidBackward)
    val (_, gradients) = caches.init.foldRight(dA -> List[(Mat, Mat, Vec)]()) {
      case ((aCur, w, b, aCache), (lastDa, grads)) =>
        val layerGrads = linearActivationBackward(lastDa, aCur, w, b, aCache, reluBackward)
        (layerGrads._1, layerGrads :: grads)
    }
    gradients ::: lastGrad :: Nil
  }

  def sigmoidBackward(dA: Mat, Z: Mat): Mat = {
    val sigm = sigmoid(Z)
    val dZ = dA *:* sigm *:* (1.0 - sigm)
    dZ
  }

  def reluBackward(dA: Mat, Z: Mat): Mat = {
    val dZ = dA.copy
    dZ(Z <:= 0.0) := 0.0
    dZ
  }

  def linearActivationBackward(dA: Mat, prevActivation: Mat, weights: Mat, biases: Vec, activationCache: Mat,
                               activationFunctionBackward: (Mat, Mat) => Mat): (Mat, Mat, Vec) = {
    val dZ = activationFunctionBackward(dA, activationCache)
    val (dAprev, dW, db) = linearBackward(dZ, prevActivation, weights, biases)
    (dAprev, dW, db)
  }

  def linearBackward(dZ: Mat, previousActivation: Mat, weights: Mat, bias: Vec): (Mat, Mat, Vec) = {
    val m = previousActivation.cols.toDouble
    val dW = (dZ * previousActivation.t) / m
    val db = sum(dZ(*, ::)) / m
    val dPrevActivation = weights.t * dZ
    (dPrevActivation, dW, db)
  }

}
