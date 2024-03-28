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
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import javax.inject.Inject

@HiltViewModel
class DetectionViewModel @Inject constructor(
    private val objectDetectorHelper: ObjectDetectorHelper
): ViewModel() {

    // State of our screen
    private var _detectionState = mutableStateOf(DetectionState())
    val detectionState: State<DetectionState> = _detectionState

    //One time event to notify the user if initialization was successful or not
    private val _tensorflowInitializationEvent = Channel<Resource<String>>()
    val tensorflowInitializationEvent = _tensorflowInitializationEvent.receiveAsFlow()

    // On Screen creation, the ViewModel will attempt to initialize the Tflite model
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
                                    _tensorflowInitializationEvent.send(Resource.Success("Tensorflow successfully initialized"))
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
                                    _tensorflowInitializationEvent.send(Resource.Error("Tensorflow failed to initialize. Error: " + detectionData.message))
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * @param image
     *      Frame from the camera
     * @param bitmapBuffer
     *      Bitmap to be passed to the ObjectDetector
     */
    fun detectObjects(image: ImageProxy, bitmapBuffer: Bitmap) {
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }

        val imageRotation = image.imageInfo.rotationDegrees
        //Passes the bitmapBuffer with the current device rotation
        val resultState = objectDetectorHelper.detect(bitmapBuffer, imageRotation)
        //Updates the state with any new detections to be used by DeviceScreen
        viewModelScope.launch(Dispatchers.Main){
            _detectionState.value = _detectionState.value.copy(
                tensorflowDetections = resultState.tensorflowDetections,
                tensorflowImageHeight = resultState.tensorflowImageHeight,
                tensorflowImageWidth = resultState.tensorflowImageWidth
            )
        }
    }

    fun destroy(){
        objectDetectorHelper.destroy()
    }
}