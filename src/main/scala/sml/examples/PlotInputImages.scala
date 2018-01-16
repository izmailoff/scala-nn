package sml.examples

import java.io.File

import breeze.linalg._
import breeze.plot._

object PlotInputImages {
  def main(args: Array[String]): Unit = {

    val f = Figure()
    val imgVec = csvread(new File("src/main/resources/data/train_x.csv"))(::, 200) // size: 1 x 12288 (r? followed by g and b)

    //  val img3channels = imgVec.toDenseMatrix.reshape(imgVec.length / 3, 3) // size: 12288/3x3 or 4096x3
    //  val img = sum(img3channels(*, ::)).toDenseMatrix // sum across each column?
    //  val greyImg = img.reshape(img.cols / 64, img.cols / 64) /:/ 3.0
    //  val p = f.subplot(0)
    //  p += image(greyImg)
    //  p.title = "Sample 0, size: " + greyImg.rows + " x " + greyImg.cols

    // simple RGB -> greyscale conversion: G =  B*0.07 + G*0.72 + R* 0.21
    val r1 = imgVec(0 until imgVec.length by 3).toDenseMatrix.reshape(64, 64) *:* 0.07 // r?
    val g1 = imgVec(1 until imgVec.length by 3).toDenseMatrix.reshape(64, 64) *:* 0.72 // g?
    val b1 = imgVec(2 until imgVec.length by 3).toDenseMatrix.reshape(64, 64) *:* 0.21 // b?
    val i1 = (r1 + g1 + b1) // /:/ 3.0
    val p1 = f.subplot(0) //(1, 2, 1)
    p1 += image(rot90(i1, 1))
    p1.title = "GREY"

    val r2 = imgVec(0 until imgVec.length by 3).toDenseMatrix.reshape(64, 64)
    val p2 = f.subplot(2, 2, 1)
    p2 += image(rot90(r2, 1))
    p2.title = "RED?"

    val g2 = imgVec(1 until imgVec.length by 3).toDenseMatrix.reshape(64, 64)
    val p3 = f.subplot(2, 2, 2)
    p3 += image(rot90(g2, 1))
    p3.title = "GREEN?"

    val b2 = imgVec(2 until imgVec.length by 3).toDenseMatrix.reshape(64, 64)
    val p4 = f.subplot(2, 2, 3)
    p4 += image(rot90(b2, 1))
    p4.title = "BLUE?"

    f.saveas("sample_image.png")
  }
}
