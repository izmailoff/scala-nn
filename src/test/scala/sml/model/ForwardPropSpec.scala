package sml.model

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest.{FlatSpec, Matchers}
import sml.helpers.TestHelpers.matricesShouldBeEqual

class ForwardPropSpec extends FlatSpec with Matchers {

  "Linear forward" should "correctly compute result" in {
    val activations = DenseMatrix(
      (1.62434536, -0.61175641),
      (-0.52817175, -1.07296862),
      (0.86540763, -2.3015387))
    val weights = DenseMatrix((1.74481176, -0.7612069, 0.3190391))
    val biases = DenseVector(-0.24937038)

    val activation = ForwardProp.linearForward(activations, weights, biases)

    matricesShouldBeEqual(activation, DenseMatrix((3.26295337, -1.23429987)))
  }

  "Sigmoid activation forward" should "correctly compute elementwise result" in {
    val prevActivations = DenseMatrix(
      (-0.41675785, -0.05626683),
      (-2.1361961, 1.64027081),
      (-1.79343559, -0.84174737))
    val weights = DenseMatrix((0.50288142, -1.24528809, -1.05795222))
    val biases = DenseVector(-0.90900761)

    val (activation, (_, _, _, cachedZ)) = ForwardProp.sigmoidActivationForward(prevActivations, weights, biases)

    matricesShouldBeEqual(activation, DenseMatrix((0.96890023, 0.11013289)))
    val activationCache = ForwardProp.linearForward(prevActivations, weights, biases)
    matricesShouldBeEqual(cachedZ, activationCache)
  }

  "Relu activation forward" should "correctly compute elementwise result" in {
    val prevActivations = DenseMatrix(
      (-0.41675785, -0.05626683),
      (-2.1361961, 1.64027081),
      (-1.79343559, -0.84174737))
    val weights = DenseMatrix((0.50288142, -1.24528809, -1.05795222))
    val biases = DenseVector(-0.90900761)

    val (activation, (_, _, _, cachedZ)) = ForwardProp.reluActivationForward(prevActivations, weights, biases)

    matricesShouldBeEqual(activation, DenseMatrix((3.43896131, 0.0)))
    val activationCache = ForwardProp.linearForward(prevActivations, weights, biases)
    matricesShouldBeEqual(cachedZ, activationCache)
  }

  "Forward propagation" should "compute correct activation and intermediate results (caches)" in {
    val input = DenseMatrix(
      (1.62434536, -0.61175641),
      (-0.52817175, -1.07296862),
      (0.86540763, -2.3015387),
      (1.74481176, -0.7612069))
    val W1 = DenseMatrix((0.3190391, -0.24937038, 1.46210794, -2.06014071),
      (-0.3224172, -0.38405435, 1.13376944, -1.09989127),
      (-0.17242821, -0.87785842, 0.04221375, 0.58281521))
    val b1 = DenseVector(-1.10061918, 1.14472371, 0.90159072)
    val W2 = DenseMatrix((0.50249434, 0.90085595, -0.68372786))
    val b2 = DenseVector(-0.12289023)
    val params = (W1, b1) :: (W2, b2) :: Nil

    val (activation, caches) = ForwardProp.forwardPropagation(input, params)

    matricesShouldBeEqual(activation, DenseMatrix((0.17007265, 0.2524272)))
    caches should have length 2
  }
}
