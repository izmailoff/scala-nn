package sml.model

import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector, csvread}
import breeze.numerics.sqrt
import breeze.stats.distributions.{Gaussian, Rand, RandBasis, ThreadLocalRandomGenerator}
import org.apache.commons.math3.random.MersenneTwister
import sml.config.GlobalAppConfig.Application.{NNMetaParams, TrainParams}
import sml.{Mat, Precision, Vec}

object Initializer {

  implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(TrainParams.randSeed)))

  def seededGaussianDistr = new Gaussian(mu = 0.0, sigma = 1.0)

  // TODO: this is a temp hack to test with predefined init params randomly generated from Python. No hardcoded paths
  // should be used when/if model save/load is implemented.
  def getInitParams() = {
    val initW1 = csvread(new File("src/main/resources/data/full_training_pass_data/W1_init_param.csv"))
    val initB1 = csvread(new File("src/main/resources/data/full_training_pass_data/b1_init_param.csv")).toDenseVector
    val initW2 = csvread(new File("src/main/resources/data/full_training_pass_data/W2_init_param.csv"))
    val initB2 = csvread(new File("src/main/resources/data/full_training_pass_data/b2_init_param.csv")).toDenseVector
    val initW3 = csvread(new File("src/main/resources/data/full_training_pass_data/W3_init_param.csv"))
    val initB3 = csvread(new File("src/main/resources/data/full_training_pass_data/b3_init_param.csv")).toDenseVector
    val initW4 = csvread(new File("src/main/resources/data/full_training_pass_data/W4_init_param.csv"))
    val initB4 = csvread(new File("src/main/resources/data/full_training_pass_data/b4_init_param.csv")).toDenseVector
    val initialParameters = (initW1, initB1) :: (initW2, initB2) :: (initW3, initB3) :: (initW4, initB4) :: Nil
    initialParameters
  }

  def getRandomParams(inputRows: Int, rand: Rand[Precision]=seededGaussianDistr) = {
    val layerDimensions = NNMetaParams.layersDimensions
    initializeParams(inputRows, layerDimensions, rand)
  }

  /**
    * Initializes all layers with random values according to provided distribution and returns them to the caller.
    *
    * @param input           Input dimension
    * @param layerDimensions optional hidden layer dimensions
    * @param rand Random distribution: gaussian, uniform, etc
    *
    * @return
    */
  def initializeParams(input: Int, layerDimensions: List[Int] = Nil, rand: Rand[Precision]=seededGaussianDistr): List[(Mat, Vec)] =
    for {
      prev :: current :: Nil <- (input :: layerDimensions).sliding(2).toList
      layer = DenseMatrix.rand[Precision](current, prev, rand) / sqrt(prev) // TODO: implement He and other init methods
      bias = DenseVector.zeros[Precision](current)
    } yield (layer, bias)
}
