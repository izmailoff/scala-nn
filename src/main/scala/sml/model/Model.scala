package sml.model

import java.util.Date

import com.typesafe.scalalogging.StrictLogging
import sml._
import sml.config.GlobalAppConfig.Application._
import sml.utils.Utils.time

object Model extends StrictLogging {
  implicit val thisLogger = logger

  def runTraining(trainSet: (Mat, Vec), testSet: (Mat, Vec)) = {
    val (xs, ys) = trainSet
    val normalizedX = time("NORMALIZE") { Normalizer.normalize(xs) }
    val parameters = time("INIT_PARAMS") {
      if(TrainParams.weightsFromFile)
        Initializer.getInitParams()
      else
        Initializer.getRandomParams(normalizedX.rows)
    }
    val layerShapes = parameters.map { case (m, v) => sml.shape(m) -> sml.shape(v) }
    logger.info(s"Initialized network with these layer dimensions: ${layerShapes.mkString(", ")}")
    val result@(params, costs, trainAccuracy, testAccuracy) = time("TRAIN") {
      train(parameters, normalizedX -> ys, NNMetaParams.learningRate, NNMetaParams.iterations, testSet)
    }
    if(TrainParams.saveAccuracyVsCostGraph) {
      val fileOut = Visualizer.plotCostVsAccuracy(costs, trainAccuracy, testAccuracy)
      logger.info(s"Saved the plot in: $fileOut.")
    }
    result
  }

  def train(parameters: List[(Mat, Vec)], trainSet: (Mat, Vec), learningRate: Double, nIterations: Int, testSet: (Mat, Vec)):
  (List[(Mat, Vec)], List[Double], List[Double], List[Double]) = {
    val (input, labels) = trainSet
    val initAccuracy = accuracy(predict(input, parameters), labels)
    logger.info(s"${new Date}: Initial accuracy on training test before training (re)started: $initAccuracy")
    val (x, y) = testSet
    val initParams = (parameters, List[Precision](), List[Precision](), List[Precision]())
    val (learnedParams, costs, trainAcc, testAcc) = (1 to nIterations).foldLeft(initParams) {
      case ((params, costs, trainAc, testAc), iter) =>
        val (cost, updatedParams) = time("FULLPATH") { Optimizer.fullPass(input, params, labels, learningRate) }
        if(iter % TrainParams.costReportIters == 0) {
          val trainPreds = predict(input, updatedParams)
          val trainAccuracy = round(accuracy(trainPreds, labels))
          val testPreds = predict(x, updatedParams)
          val testAccuracy = round(accuracy(testPreds, y))
          val costDiff = cost - costs.headOption.getOrElse(0.0)
          logger.info(s"${new Date}: Iteration $iter/$nIterations finished. Cost: $cost. Cost changed by: $costDiff." +
            s" Train-set Accuracy: $trainAccuracy, Test-set Accuracy: $testAccuracy")
          (updatedParams, cost :: costs, trainAccuracy :: trainAc, testAccuracy :: testAc)
        }
        else {
          (updatedParams, costs, trainAc, testAc)
        }
      // TODO: Checkpoint function, early stopping, etc
    }
    (learnedParams, costs.reverse, trainAcc.reverse, testAcc.reverse)
  }

  def predict(X: Mat, params: List[(Mat, Vec)]): Vec = {
    val (probas, _) = ForwardProp.forwardPropagation(X, params)
    probas(probas <:= 0.5) := 0.0
    probas(probas >:> 0.5) := 1.0
    probas.toDenseVector
  }

  def accuracy(predictions: Vec, trueLabels: Vec): Double =
    (predictions :== trueLabels).activeSize.toDouble / predictions.length

  def round(x: Double, scale: Int = 4): Double =
    BigDecimal(x).setScale(scale, BigDecimal.RoundingMode.HALF_UP).toDouble
}
