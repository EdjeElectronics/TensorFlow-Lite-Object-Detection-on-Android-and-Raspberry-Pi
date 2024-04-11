package io.ejtech.tflite.ui.detection

/**
 * @param label
 *      Name of the detection
 * @param confidence_score
 */
data class CategoryMut (
    var label: String = "",
    var confidence_score: Float = 0.0f,
)