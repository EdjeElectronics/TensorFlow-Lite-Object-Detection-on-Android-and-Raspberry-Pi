package io.ejtech.tflite.ui.detection

import android.graphics.RectF

data class DetectionMut(
    var boundingBox: RectF,
    var category: CategoryMut
)
