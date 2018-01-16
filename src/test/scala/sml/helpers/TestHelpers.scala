package sml.helpers

import sml.{Mat, Vec}
import sml.shape
import breeze.numerics._
import org.scalatest.{Assertion, Matchers}

object TestHelpers extends Matchers {

  def matricesEqual(m1: Mat, m2: Mat, thresh: Double = 1e-7d): Boolean = {
    breeze.linalg.all(abs(m1 - m2) <:< thresh)
  }

  def matricesShouldBeEqual(actual: Mat, expected: Mat, thresh: Double = 1e-7d): Assertion = {
    def msg = s"\nactual:\n$actual\nwith shape ${shape(actual)}.\n\nexpected:\n$expected\nwith shape ${shape(expected)}.\n\nthreshold: $thresh.\n"
    withClue(s"Rows don't match!\n$msg\n") { actual.rows should be(expected.rows) }
    withClue(s"Columns don't match!\n$msg\n") { actual.cols should be(expected.cols) }
    assert(matricesEqual(actual, expected), s"Values don't match!\n$msg\nactual - expected:\n${actual - expected}\n")
  }

  def vectorsEqual(v1: Vec, v2: Vec, thresh: Double = 1e-7d): Boolean = {
    breeze.linalg.all(abs(v1 - v2) <:< thresh)
  }

  // TODO: update this to match matricesShouldBeEqual
  def vectorsShouldBeEqual(v1: Vec, v2: Vec, thresh: Double = 1e-7d): Assertion = {
    def msg = s"v1: $v1 with shape ${shape(v1)}; v2: $v2 with shape ${shape(v2)}; threshold: $thresh."
    withClue(s"$msg Length doesn't match!") { v1.length should be(v2.length) }
    assert(vectorsEqual(v1, v2), s"$msg v1 - v2: ${v1 - v2}.")
  }

}
