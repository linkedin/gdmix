package com.linkedin.gdmix.utils

import com.fasterxml.jackson.annotation.JsonInclude.Include
import com.fasterxml.jackson.databind.{DeserializationFeature, ObjectMapper, SerializationFeature}
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper

/**
 * Helper class to serialize an Object into a JSON String and to deserialize a Json String into a Map
 */
object JsonUtils {

  /**
   * Define the FasterXML Jackson object mapper
   */
  private[this] val mapper = new ObjectMapper() with ScalaObjectMapper
  mapper
    .registerModule(DefaultScalaModule)
    .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
    .configure(DeserializationFeature.ACCEPT_SINGLE_VALUE_AS_ARRAY, true)
    .configure(SerializationFeature.INDENT_OUTPUT, true)
    .setSerializationInclusion(Include.NON_NULL)

  /**
   * Write an object to JSON formatted String
   *
   * @param value Any object to be written
   * @return A JSON pretty formatted String
   */
  def toJsonString(value: Any): String = mapper.writeValueAsString(value)

  /**
   * Helper function to generate a map from json string
   */
  def toMap[V](json: String)(implicit m: Manifest[V]): Map[String, V] = {
    if(json.trim.isEmpty)
      Map.empty[String, V]

    fromJson[Map[String, V]](json)
  }

  private def fromJson[T](json: String)(implicit m: Manifest[T]): T = {
    mapper.readValue[T](json)
  }
}
