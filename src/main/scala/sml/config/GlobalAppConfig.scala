package sml.config

import com.typesafe.config.ConfigFactory

import scala.collection.JavaConverters._

object GlobalAppConfig {

  val config = ConfigFactory.load()

  object Application {

    object NNMetaParams {
      private lazy val metaParmsConf = config.getConfig("application.nnMetaParams")
      lazy val iterations = metaParmsConf.getInt("iterations")
      lazy val learningRate = metaParmsConf.getDouble("learningRate")
      lazy val layersDimensions: List[Int] = metaParmsConf.getIntList("layersDimensions").asScala.toList.map(_.toInt)
      lazy val layersStr = s"[${layersDimensions.mkString("x")}]"
    }

    object TrainParams {
      private lazy val trainParmsConf = config.getConfig("application.trainParams")
      // TODO: implement popular format loading like h5 and pass file paths, for now those are hardcoded.
      lazy val weightsFromFile = trainParmsConf.getBoolean("weightsFromFile")
      lazy val randSeed = trainParmsConf.getInt("randSeed")
      lazy val costReportIters = trainParmsConf.getInt("costReportIters")
      lazy val saveAccuracyVsCostGraph = trainParmsConf.getBoolean("saveAccuracyVsCostGraph")
    }

  }

}

