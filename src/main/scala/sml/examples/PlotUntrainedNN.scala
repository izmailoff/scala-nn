package sml.examples

import breeze.numerics._
import breeze.plot._
import breeze.stats.distributions.Rand
import breeze.stats.meanAndVariance
import sml.model.Initializer
import sml.{Mat, Vec}

object PlotUntrainedNN extends App {

  val totalRows = 3

  val f = Figure()
  val f2 = Figure()
  val f3 = Figure()

  val originalModelInputSize = 12288
  val cutoffSize = 1000
  val randGaussianParams = Initializer.getRandomParams(cutoffSize, Rand.gaussian)
  val randUniformParams = Initializer.getRandomParams(cutoffSize, Rand.uniform)
  val initParams = Initializer.getInitParams().map{case (w, b) =>
    (if(w.rows > cutoffSize) w(1 to cutoffSize, ::)
    else if (w.cols > cutoffSize) w(::, 1 to cutoffSize)
    else w) -> b
  }

  def plotWeightsImg(params: List[(Mat, Vec)], row: Int, preLabel: String): Unit = {
    val weights = params.map{case (w, _) => w}
    val nWeights = weights.length
    for {
      (w, i) <- weights.zipWithIndex
    } {
      val p = f.subplot(totalRows, nWeights, i + (nWeights * (row - 1)))
      val wSide = sqrt(w.size).toInt
      val wSquare = w.toDenseVector(1 to wSide*wSide).toDenseMatrix.reshape(wSide, wSide)
      p += image(wSquare)
      val stats = meanAndVariance(w) // TODO: compute once
      val mean = f"${stats.mean}%1.10f"
      val variance = f"${stats.variance}%1.10f"
      p.title = s"$preLabel: W${i+1}, mean: $mean, var: $variance, cnt: ${stats.count}"
    }
  }

  plotWeightsImg(randGaussianParams, 1, "Gaussian")
  plotWeightsImg(randUniformParams, 2, "Uniform")
  plotWeightsImg(initParams, 3, "InitParams")

  def plotWeightsGraph(params: List[(Mat, Vec)], row: Int, preLabel: String): Unit = {
    val weights = params.map{case (w, _) => w}
    val nWeights = weights.length
    for {
      (w, i) <- weights.zipWithIndex
    } {
      val p = f2.subplot(totalRows, nWeights, i + (nWeights * (row - 1)))
      val Y = w.toArray
      val X = (0 until Y.length).map(_.toDouble)
      //p += scatter(X, Y, _ => 1.0)
      p += plot(X, Y, lines = false, shapes = true, style='.')
      val stats = meanAndVariance(w)
      val mean = f"${stats.mean}%1.10f"
      val variance = f"${stats.variance}%1.10f"
      p.title = s"$preLabel: W${i+1}, mean: $mean, var: $variance, cnt: ${stats.count}"
    }
  }

  plotWeightsGraph(randGaussianParams, 1, "Gaussian")
  plotWeightsGraph(randUniformParams, 2, "Uniform")
  plotWeightsGraph(initParams, 3, "InitParams")

  def plotWeightsHist(params: List[(Mat, Vec)], row: Int, preLabel: String): Unit = {
    val weights = params.map{case (w, _) => w}
    val nWeights = weights.length
    for {
      (w, i) <- weights.zipWithIndex
    } {
      val p = f3.subplot(totalRows, nWeights, i + (nWeights * (row - 1)))
      val Y = w.toDenseVector
      p += hist(Y, 1000)
      val stats = meanAndVariance(w)
      val mean = f"${stats.mean}%1.10f"
      val variance = f"${stats.variance}%1.10f"
      p.title = s"$preLabel: W${i+1}, mean: $mean, var: $variance, cnt: ${stats.count}"
    }
  }

  plotWeightsHist(randGaussianParams, 1, "Gaussian")
  plotWeightsHist(randUniformParams, 2, "Uniform")
  plotWeightsHist(initParams, 3, "InitParams")

}
