package sml.model

import breeze.linalg.{*, sum}
import breeze.numerics.log
import com.typesafe.scalalogging.StrictLogging
import sml.examples.ModelTrain.logger
import sml.utils.Utils.time
import sml.{Mat, Vec}

object Optimizer extends StrictLogging {
  implicit val thisLogger = logger

  def fullPass(input: Mat, params: List[(Mat, Vec)], labels: Vec, learningRate: Double): (Double, List[(Mat, Vec)]) = {
    val (lastActivation, caches) = time("FORWARD_PROP") { ForwardProp.forwardPropagation(input, params) }
    val cost = time("COST") { computeCost(lastActivation, labels) }
    val activationsAndGrads = time("BACK_PROP") { BackProp.backwardPropagation(lastActivation, labels, caches) }
    val grads = activationsAndGrads.map { case (da, dw, db) => dw -> db }
    val updatedParams = time("UPDATE_PARAMS") { updateParameters(params, grads, learningRate) }
    cost -> updatedParams
  }

  def computeCost(lastActivation: Mat, labels: Vec): Double = {
    val logA1 = log(lastActivation).t
    val pred1 = logA1(::, *) *:* labels
    val logA0 = log(1.0 - lastActivation).t
    val pred0 = logA0(::, *) *:* (1.0 - labels)
    val cost = -sum(pred1 + pred0) / labels.length
    cost
  }

  def updateParameters(params: List[(Mat, Vec)], grads: List[(Mat, Vec)], learningRate: Double): List[(Mat, Vec)] =
    for {
      ((w, b), (dw, db)) <- params.zip(grads)
      updatedW = w - (learningRate * dw)
      updatedB = b - (learningRate * db)
    } yield updatedW -> updatedB
}
