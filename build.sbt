name := "scala-nn"

version := "0.1"
     
scalaVersion := "2.12.3"

javacOptions ++= Seq("-target", "1.8")

scalacOptions in Test ++= Seq("-Yrangepos")

resolvers ++= Seq(
  "Sonatype Snapshots" at "http://oss.sonatype.org/content/repositories/snapshots",
  "Sonatype Releases" at "http://oss.sonatype.org/content/repositories/releases"
)

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.13.2",
  // native libraries are not included by default. add this if you want them (as of 0.7)
  // native libraries greatly improve performance, but increase jar sizes.
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  // the visualization library is distributed separately as well.
  // It depends on LGPL code.
  "org.scalanlp" %% "breeze-viz" % "0.13.2",

  //"com.github.fommil.netlib" % "all" % "1.1.2" pomOnly(), // ??? FIXME: this is probably already included

  // config:
  "com.typesafe" % "config" % "1.3.2",

  // logging:
  "ch.qos.logback" % "logback-classic" % "1.2.3",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.7.2",

  "org.scalactic" %% "scalactic" % "3.0.1",
  "org.scalatest" %% "scalatest" % "3.0.1" % "test"
)

mainClass in assembly := Some("sml.examples.ModelTrain")
