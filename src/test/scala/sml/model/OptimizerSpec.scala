package sml.model

import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector, csvread}
import org.scalatest.{FlatSpec, Matchers}
import sml.helpers.TestHelpers.{matricesShouldBeEqual, vectorsShouldBeEqual}

class OptimizerSpec extends FlatSpec with Matchers {

  "Cost function" should "compute correct cost given last layer activations and labels" in {
    val lastActivation = DenseMatrix((0.8, 0.9, 0.4))
    val labels = DenseVector(1.0, 1.0, 1.0)

    val cost = Optimizer.computeCost(lastActivation, labels)

    cost should equal(0.414931599615 +- 1e-12d)
  }

  "Update parameters" should "correctly create new parameters based on learning rate and gradients" in {
    val W1 = DenseMatrix((-0.41675785, -0.05626683, -2.1361961, 1.64027081),
      (-1.79343559, -0.84174737, 0.50288142, -1.24528809),
      (-1.05795222, -0.90900761, 0.55145404, 2.29220801))
    val b1 = DenseVector(0.04153939, -1.11792545, 0.53905832) // it looked like transposed / column vector
    val W2 = DenseMatrix((-0.5961597, -0.0191305, 1.17500122))
    val b2 = DenseVector(-0.74787095)
    val params = (W1, b1) :: (W2, b2) :: Nil
    val dW1 = DenseMatrix((1.78862847, 0.43650985, 0.09649747, -1.8634927),
      (-0.2773882, -0.35475898, -0.08274148, -0.62700068),
      (-0.04381817, -0.47721803, -1.31386475, 0.88462238)) // it looked like transposed / column vector
    val db1 = DenseVector(0.88131804, 1.70957306, 0.05003364)
    val dW2 = DenseMatrix((-0.40467741, -0.54535995, -1.54647732))
    val db2 = DenseVector(0.98236743)
    val grads = (dW1, db1) :: (dW2, db2) :: Nil

    val (updatedW1, updatedb1) :: (updatedW2, updatedb2) :: Nil = Optimizer.updateParameters(params, grads, learningRate = 0.1)

    matricesShouldBeEqual(updatedW1,
      DenseMatrix((-0.59562069, -0.09991781, -2.14584584, 1.82662008),
        (-1.76569676, -0.80627147, 0.51115557, -1.18258802),
        (-1.0535704, -0.86128581, 0.68284052, 2.20374577)))
    vectorsShouldBeEqual(updatedb1, DenseVector(-0.04659241, -1.28888275, 0.53405496))
    matricesShouldBeEqual(updatedW2, DenseMatrix((-0.55569196, 0.0354055, 1.32964895)))
    vectorsShouldBeEqual(updatedb2, DenseVector(-0.84610769))
  }

  "Full pass: forward <-> backward" should "produce correct trained weights and biases" in {
    val (trainX, trainY) = Serializer.readXYinputs("src/main/resources/data/train_x.csv", "src/main/resources/data/train_y.csv")
    val normalizedTrainX = Normalizer.normalize(trainX)
    //val layerDimensions = List(20, 7, 5, 1)
    val learningRate = 0.0075
    val initW1 = csvread(new File("src/main/resources/data/full_training_pass_data/W1_init_param.csv"))
    val initB1 = csvread(new File("src/main/resources/data/full_training_pass_data/b1_init_param.csv")).toDenseVector
    val initW2 = csvread(new File("src/main/resources/data/full_training_pass_data/W2_init_param.csv"))
    val initB2 = csvread(new File("src/main/resources/data/full_training_pass_data/b2_init_param.csv")).toDenseVector
    val initW3 = csvread(new File("src/main/resources/data/full_training_pass_data/W3_init_param.csv"))
    val initB3 = csvread(new File("src/main/resources/data/full_training_pass_data/b3_init_param.csv")).toDenseVector
    val initW4 = csvread(new File("src/main/resources/data/full_training_pass_data/W4_init_param.csv"))
    val initB4 = csvread(new File("src/main/resources/data/full_training_pass_data/b4_init_param.csv")).toDenseVector
    val initialParameters = (initW1, initB1) :: (initW2, initB2) :: (initW3, initB3) :: (initW4, initB4) :: Nil
    //val (testX, testY) = Model.readXYinputs("src/main/resources/data/test_x.csv", "src/main/resources/data/test_y.csv")
    val expectedW1 = csvread(new File("src/main/resources/data/full_training_pass_data/W1_param.csv"))
    val expectedB1 = csvread(new File("src/main/resources/data/full_training_pass_data/b1_param.csv")).toDenseVector
    val expectedW2 = csvread(new File("src/main/resources/data/full_training_pass_data/W2_param.csv"))
    val expectedB2 = csvread(new File("src/main/resources/data/full_training_pass_data/b2_param.csv")).toDenseVector
    val expectedW3 = csvread(new File("src/main/resources/data/full_training_pass_data/W3_param.csv"))
    val expectedB3 = csvread(new File("src/main/resources/data/full_training_pass_data/b3_param.csv")).toDenseVector
    val expectedW4 = csvread(new File("src/main/resources/data/full_training_pass_data/W4_param.csv"))
    val expectedB4 = csvread(new File("src/main/resources/data/full_training_pass_data/b4_param.csv")).toDenseVector

    val (cost, params) = Optimizer.fullPass(normalizedTrainX, initialParameters, trainY, learningRate)

    cost should equal(0.771749 +- 1e-6d)
    params should have length 4
    val (w1, b1) :: (w2, b2) :: (w3, b3) :: (w4, b4) :: Nil = params
    matricesShouldBeEqual(w1, expectedW1)
    vectorsShouldBeEqual(b1, expectedB1)
    matricesShouldBeEqual(w2, expectedW2)
    vectorsShouldBeEqual(b2, expectedB2)
    matricesShouldBeEqual(w3, expectedW3)
    vectorsShouldBeEqual(b3, expectedB3)
    matricesShouldBeEqual(w4, expectedW4)
    vectorsShouldBeEqual(b4, expectedB4)

    // TODO: use testX for predict
  }

}
