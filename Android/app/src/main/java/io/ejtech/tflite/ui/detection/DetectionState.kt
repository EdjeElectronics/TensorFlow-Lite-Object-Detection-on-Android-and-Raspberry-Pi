package io.ejtech.tflite.ui.detection

import org.tensorflow.lite.task.gms.vision.detector.Detection

/**
 * Used by DetectionScreen to update itself from DetectionViewMdeol
 */
data class DetectionState(
    var tensorflowEnabled: Boolean = false,
    var tensorflowDetections: MutableList<Detection> = mutableListOf(),
    var tensorflowImageHeight: Int = 0,
    var tensorflowImageWidth: Int = 0,
)
