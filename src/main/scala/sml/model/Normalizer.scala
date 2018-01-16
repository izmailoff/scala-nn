package sml.model

import sml.Mat

object Normalizer {

  // TODO: implement different normalization options. This is just a quick hack
  def normalize(data: Mat): Mat =
    data / 255.0
}
