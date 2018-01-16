package sml.utils

import java.util.concurrent.TimeUnit

import com.typesafe.scalalogging.Logger

object Utils {

  def time[R](label: String)(block: => R)(implicit logger: Logger): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    val elapsed = TimeUnit.NANOSECONDS.toMillis(t1 - t0)
    logger.debug(s"$label: Elapsed time: $elapsed ms.")
    result
  }

}
