package sml.examples

import com.typesafe.scalalogging.StrictLogging
import sml.model.{Model, Serializer}
import sml.utils.Utils.time


object ModelTrain extends App with StrictLogging {
  implicit val thisLogger = logger

  val trainSet = time("LOAD_TRAIN_DATA") {
    Serializer.readXYinputs("src/main/resources/data/train_x.csv", "src/main/resources/data/train_y.csv") }
  val testSet = time("LOAD_TEST_DATA") {
    Serializer.readXYinputs("src/main/resources/data/test_x.csv", "src/main/resources/data/test_y.csv") }
  val (params, _, _, _) = time("TRAINING"){ Model.runTraining(trainSet, testSet) }
  val (x, y) = testSet
  val predictions = time("PREDICT_TEST"){ Model.predict(x, params) }
  val accuracy = Model.accuracy(predictions, y)
  logger.info(s"ACCURACY: $accuracy")
}
