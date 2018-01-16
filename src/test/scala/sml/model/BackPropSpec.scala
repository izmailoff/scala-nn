package sml.model

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest.{FlatSpec, Matchers}
import sml.helpers.TestHelpers.{matricesShouldBeEqual, vectorsShouldBeEqual}
import sml.shape

class BackPropSpec extends FlatSpec with Matchers {

  "Linear backward" should "correctly compute gradients" in {
    val dZ = DenseMatrix((1.62434536, -0.61175641))
    val prevA = DenseMatrix(
      (-0.52817175, -1.07296862),
      (0.86540763, -2.3015387),
      (1.74481176, -0.7612069))
    val W = DenseMatrix((0.3190391, -0.24937038, 1.46210794))
    val b = DenseVector(-2.06014071)

    val (dPrevA, dW, db) = BackProp.linearBackward(dZ, prevA, W, b)

    shape(dPrevA) should be(shape(prevA))
    shape(dW) should be(shape(W))
    shape(db) should be(shape(b))
    matricesShouldBeEqual(dPrevA,
      DenseMatrix((0.51822968, -0.19517421),
        (-0.40506361, 0.15255393),
        (2.37496825, -0.89445391)))
    matricesShouldBeEqual(dW, DenseMatrix((-0.10076895, 1.40685096, 1.64992505)))
    vectorsShouldBeEqual(db, DenseVector(0.50629448))
  }

  "Sigmoid backward" should "correctly compute activation gradient" in {
    val dA = DenseMatrix((-0.41675785, -0.05626683))
    val Z = DenseMatrix((0.04153939, -1.11792545))

    val dZ = BackProp.sigmoidBackward(dA, Z)

    shape(dZ) should be(shape(Z))
    // TODO: add value assertions
  }

  "Relu backward" should "correctly compute activation gradient" in {
    val dA = DenseMatrix((-0.41675785, -0.05626683))
    val Z = DenseMatrix((0.04153939, -1.11792545))

    val dZ = BackProp.reluBackward(dA, Z)

    shape(dZ) should be(shape(Z))
    // TODO: add value assertions
  }

  "Sigmoid Activation backward" should "correctly compute gradients" in {
    val AL = DenseMatrix((-0.41675785, -0.05626683))
    val prevA = DenseMatrix(
      (-2.1361961, 1.64027081),
      (-1.79343559, -0.84174737),
      (0.50288142, -1.24528809))
    val W = DenseMatrix((-1.05795222, -0.90900761, 0.55145404))
    val b = DenseVector(2.29220801)
    val activationCache = DenseMatrix((0.04153939, -1.11792545))

    val (dAprev, dW, db) = BackProp.linearActivationBackward(AL, prevA, W, b, activationCache, BackProp.sigmoidBackward)

    matricesShouldBeEqual(dAprev,
      DenseMatrix((0.11017994, 0.01105339),
        (0.09466817, 0.00949723),
        (-0.05743092, -0.00576154)))
    matricesShouldBeEqual(dW, DenseMatrix((0.10266786, 0.09778551, -0.01968084)))
    vectorsShouldBeEqual(db, DenseVector(-0.05729622))
  }

  "Relu Activation backward" should "correctly compute gradients" in {
    val AL = DenseMatrix((-0.41675785, -0.05626683))
    val prevA = DenseMatrix(
      (-2.1361961, 1.64027081),
      (-1.79343559, -0.84174737),
      (0.50288142, -1.24528809))
    val W = DenseMatrix((-1.05795222, -0.90900761, 0.55145404))
    val b = DenseVector(2.29220801)
    val activationCache = DenseMatrix((0.04153939, -1.11792545))

    val (dAprev, dW, db) = BackProp.linearActivationBackward(AL, prevA, W, b, activationCache, BackProp.reluBackward)

    matricesShouldBeEqual(dAprev,
      DenseMatrix((0.44090989, 0.0),
        (0.37883606, 0.0),
        (-0.2298228, 0.0)))
    matricesShouldBeEqual(dW, DenseMatrix((0.44513824, 0.37371418, -0.10478989)))
    vectorsShouldBeEqual(db, DenseVector(-0.20837892))
  }

  "Backward propagation" should "correctly compute all gradients" in {
    val AL = DenseMatrix((1.78862847, 0.43650985))
    val Y = DenseVector(1.0, 0.0) // Yassess
    val caches = List(
      (
        DenseMatrix((0.09649747, -1.8634927),
          (-0.2773882, -0.35475898),
          (-0.08274148, -0.62700068),
          (-0.04381817, -0.47721803)),
        DenseMatrix((-1.31386475, 0.88462238, 0.88131804, 1.70957306),
          (0.05003364, -0.40467741, -0.54535995, -1.54647732),
          (0.98236743, -1.10106763, -1.18504653, -0.2056499)),
        DenseVector(1.48614836, 0.23671627, -1.02378514),
        DenseMatrix((-0.7129932, 0.62524497),
          (-0.16051336, -0.76883635),
          (-0.23003072, 0.74505627))
      ),
      (
        DenseMatrix((1.97611078, -1.24412333),
          (-0.62641691, -0.80376609),
          (-2.41908317, -0.92379202)),
        DenseMatrix((-1.02387576, 1.12397796, -0.13191423)),
        DenseVector(-1.62328545),
        DenseMatrix((0.64667545, -0.35627076))
      )
    )

    val gradients = BackProp.backwardPropagation(AL, Y, caches)

    gradients should have length 2
    val (dA1, dW1, db1) :: _ :: Nil = gradients
    matricesShouldBeEqual(dW1,
      DenseMatrix((0.41010002, 0.07807203, 0.13798444, 0.10502167),
        (0.0, 0.0, 0.0, 0.0),
        (0.05283652, 0.01005865, 0.01777766, 0.0135308)))
    vectorsShouldBeEqual(db1, DenseVector(-0.22007063, 0.0, -0.02835349))
    matricesShouldBeEqual(dA1,
      DenseMatrix((0.0, 0.52257901),
        (0.0, -0.3269206),
        (0.0, -0.32070404),
        (0.0, -0.74079187)))
    // TODO: add assertions for second layer gradient's values
  }

}
