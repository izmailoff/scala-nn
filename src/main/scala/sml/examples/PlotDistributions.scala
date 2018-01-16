package sml.examples

import breeze.stats.distributions.Rand

object PlotDistributions {

  def main(args: Array[String]) = {
    import breeze.linalg._
    import breeze.plot._

    val n = 100
    val f = Figure()

    val p = f.subplot(0)
    val g = DenseVector.rand[Double](n * n, Rand.gaussian) //* 0.01
    p += hist(g, 100)
    p.title = "Normal/gaussian distribution"

    val p2 = f.subplot(2,2,1)
    p2 += image(g.toDenseMatrix.reshape(n, n))
    p2.title = "Normal/gaussian distribution"

    val p3 = f.subplot(2,2,2)
    val g3 = DenseVector.rand[Double](n * n, Rand.uniform) //* 0.01
    p3 += hist(g3,100)
    p3.title = "Uniform distribution"

    val p4 = f.subplot(2,2,3)
    p4 += image(g3.toDenseMatrix.reshape(n, n))
    p4.title = "Uniform distribution"

    f.saveas("image.png")

  }

}