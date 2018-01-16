package sml.model

import org.scalatest.{FlatSpec, Matchers}

class SerializerSpec extends FlatSpec with Matchers {

  "Input load from files" should "correctly read data from CSV" in {
    val (xs, ys) = Serializer.readXYinputs("src/main/resources/data/train_x.csv", "src/main/resources/data/train_y.csv") // use filepath separator

    xs.rows should be(12288)
    xs.cols should be(209)
    ys.length should be(209)
  }
}
