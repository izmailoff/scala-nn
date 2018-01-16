package sml.model

import org.scalatest.{FlatSpec, Matchers}

class InitializerSpec extends FlatSpec with Matchers {

  "Initialize parameters" should "create corresponding matrices and vectors" in {
    val parameters = Initializer.initializeParams(5, List(4, 3))

    parameters should have length 2
    val (w1, b1) :: (w2, b2) :: Nil = parameters
    w1.rows should be(4)
    w1.cols should be(5)
    b1.length should be(4)
    w2.rows should be(3)
    w2.cols should be(4)
    b2.length should be(3)
  }
}
