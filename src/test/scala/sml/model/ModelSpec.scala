package sml.model

import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector, csvread}
import org.scalatest._
import sml.helpers.TestHelpers._
import sml._
import sml.utils.Utils.time

class ModelSpec extends FlatSpec with Matchers {

  ignore /*"Train model"*/ should "????? make this a proper test later" in {
    val layerDims = ???
    val X = ???
    val Y = ???
    val alpha = ???
    val nIter = ???

    val (params, costs, trainAccuracy, testAccuracy) = Model.train(layerDims, X, Y, alpha, nIter)

    // TODO: do assertions
  }

  ignore /*"Predict"*/ should "correctly classify ???" in {
    val X = ??? // TODO: load from csv
    val Y = ???
    val params = ???

    val predictions = Model.predict(X, params)

    // TODO: do assertions.
  }

  // TODO: add test for accuracy
}
