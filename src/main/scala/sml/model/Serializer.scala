package sml.model

import java.io.File

import breeze.linalg.csvread
import sml.{Mat, Vec}

object Serializer {

  def readXYinputs(filenameX: String, filenameY: String): (Mat, Vec) = {
    val X = csvread(new File(filenameX)) // TODO: library has implicits to avoid File
    val Y = csvread(new File(filenameY)).toDenseVector
    X -> Y
  }
}
