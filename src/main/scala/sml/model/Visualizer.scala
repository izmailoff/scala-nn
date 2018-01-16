package sml.model

import sml.config.GlobalAppConfig.Application.NNMetaParams._
import sml.config.GlobalAppConfig.Application.TrainParams._

object Visualizer {

  def plotCostVsAccuracy(cost: List[Double], trainSetAccuracy: List[Double], testSetAccuracy: List[Double]): String = {
    import breeze.plot._
    val X = (0 until cost.length).map(_ * costReportIters.toDouble)
    val f = Figure()
    f.visible = false // for headless mode: java -Djava.awt.headless=true, See: https://github.com/scalanlp/breeze-viz/issues/9
    f.width = 1920
    f.height = 1080
    val p = f.subplot(0)
    p += plot(X, cost, name="Cost")
    p.legend = true
    p.setYAxisDecimalTickUnits()
    p.title = "Cost."
    val p2 = f.subplot(2, 1, 1)
    p2 += plot(X, trainSetAccuracy, name="Train accuracy")
    p2 += plot(X, testSetAccuracy, name="Test accuracy")
    p2.legend = true
    p2.setYAxisDecimalTickUnits()
    p2.title = "Training Accuracy/Test Accuracy comparison."
    val sinceEpoch = System.currentTimeMillis / 1000
    val filename = s"${sinceEpoch}_cost_and_accuracy_${layersStr}_${iterations}_${learningRate}.png"
    f.saveas(filename)
    filename
  }
}
