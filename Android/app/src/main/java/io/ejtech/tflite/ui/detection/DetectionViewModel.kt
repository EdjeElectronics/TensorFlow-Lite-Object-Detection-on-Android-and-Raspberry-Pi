package io.ejtech.tflite.ui.detection

import android.graphics.Bitmap
import android.graphics.RectF
import androidx.camera.core.ImageProxy
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import javax.inject.Inject

@HiltViewModel
class DetectionViewModel @Inject constructor(
    private val objectDetectorHelper: ObjectDetectorHelper
): ViewModel() {

    private var _detectionState = mutableStateOf(DetectionState())
    val detectionState: State<DetectionState> = _detectionState

    init {
        if(!detectionState.value.tensorflowEnabled){
            viewModelScope.launch(Dispatchers.IO) {
                objectDetectorHelper.initialize().collect { detectionData: Resource<DetectionState> ->
                    when(detectionData){
                        is Resource.Success -> {
                            detectionData.data?.let{
                                withContext(Dispatchers.Main){
                                    _detectionState.value = _detectionState.value.copy(
                                        tensorflowEnabled = it.tensorflowEnabled
                                    )
                                }
                            }
                        }
                        is Resource.Loading -> {}
                        is Resource.Error -> {
                            detectionData.data?.let{
                                withContext(Dispatchers.Main){
                                    _detectionState.value = _detectionState.value.copy(
                                        tensorflowEnabled = it.tensorflowEnabled
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fun detectObjects(image: ImageProxy, bitmapBuffer: Bitmap) {
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }

        val imageRotation = image.imageInfo.rotationDegrees
        val resultState = objectDetectorHelper.detect(bitmapBuffer, imageRotation)

        viewModelScope.launch(Dispatchers.Main){
            _detectionState.value = _detectionState.value.copy(
                tensorflowDetections = resultState.tensorflowDetections,
                tensorflowImageHeight = resultState.tensorflowImageHeight,
                tensorflowImageWidth = resultState.tensorflowImageWidth
            )
        }
    }
}