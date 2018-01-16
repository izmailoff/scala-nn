import breeze.linalg.{DenseMatrix, DenseVector}

package object sml {

  // Define precision for the whole application in one place. It can be changed here from Double to Float for instance.
  type Precision = Double
  type Mat = DenseMatrix[Precision]
  type Vec = DenseVector[Precision]
  //type Labels = DenseVector[Int] // TODO: need this?

  /**
    * Numpy-like shape function that gives a tensor dimensions as a tuple.
    * @param matrix
    * @return dimensions
    */
  def shape(matrix: Mat): (Int, Int) = (matrix.rows, matrix.cols)

  /**
    * Numpy-like shape function that gives a tensor dimensions as a tuple.
    * @param vector
    * @return dimensions
    */
  def shape(vector: Vec): Int = vector.length
}
