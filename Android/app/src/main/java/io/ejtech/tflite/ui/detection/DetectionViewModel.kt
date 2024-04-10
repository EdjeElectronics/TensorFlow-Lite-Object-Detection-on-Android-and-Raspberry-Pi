package io.ejtech.tflite.ui.detection

import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.geometry.Size
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import javax.inject.Inject
import kotlin.math.abs
import kotlin.math.max

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

    private var avgInference: MutableList<Long> = mutableListOf()

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
    fun detectObjects(image: ImageProxy, bitmapBuffer: Bitmap, boxsize: Size) {
        var inferenceTime = SystemClock.uptimeMillis()
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }

        val imageRotation = image.imageInfo.rotationDegrees
        //Passes the bitmapBuffer with the current device rotation
        val resultState = objectDetectorHelper.detect(bitmapBuffer, imageRotation)
        //Updates the state with any new detections to be used by DeviceScreen

        //Remove underscores
        resultState.tensorflowDetections.forEach { it.category.label = it.category.label.replace("_", " ") }
        //Filter vehicle
        resultState.tensorflowDetections.removeAll{it.category.label == "vehicle"}
        //Update wheel names based on closeness to other parts
        val wheels = resultState.tensorflowDetections.filter {it.category.label == "wheel"}
        if(wheels.isNotEmpty()){
            wheels.forEach{ part ->
                val idx = resultState.tensorflowDetections.indexOf(part)
                val wheelleft = part.boundingBox.left
                var partclosesttowheelleft = DetectionMut(RectF(99999f, 99999f, 99999f, 99999f), CategoryMut())
                var diff = abs(wheelleft - partclosesttowheelleft.boundingBox.left)
                resultState.tensorflowDetections.forEach {
                    if (it.category.label != "wheel") {
                        if (abs(wheelleft - it.boundingBox.left) < diff) {
                            partclosesttowheelleft = it
                            diff = abs(wheelleft - it.boundingBox.left)
                        }
                    }
                }
                //Just to check that we found detections first
                if(partclosesttowheelleft.category.label != "") {
                    val partclosestowheelleftlabel = partclosesttowheelleft.category.label
                    if (partclosestowheelleftlabel == "left headlamp" ||
                        partclosestowheelleftlabel == "left fender" ||
                        partclosestowheelleftlabel == "left front door"
                    ) {
                        resultState.tensorflowDetections[idx].category.label = "left front wheel"
                    } else if (partclosestowheelleftlabel == "right headlamp" ||
                        partclosestowheelleftlabel == "right fender" ||
                        partclosestowheelleftlabel == "right front door"
                    ) {
                        resultState.tensorflowDetections[idx].category.label = "right front wheel"
                    } else if (partclosestowheelleftlabel == "left rear door" ||
                        partclosestowheelleftlabel == "left quarter panel" ||
                        partclosestowheelleftlabel == "left tail lamp"
                    ) {
                        resultState.tensorflowDetections[idx].category.label = "left rear wheel"
                    } else if (partclosestowheelleftlabel == "right rear door" ||
                        partclosestowheelleftlabel == "right quarter panel" ||
                        partclosestowheelleftlabel == "right tail lamp"
                    ) {
                        resultState.tensorflowDetections[idx].category.label = "right rear wheel"
                    } else if (partclosestowheelleftlabel == "front bumper cover") {
                        if (partclosesttowheelleft.boundingBox.left < wheelleft) {
                            resultState.tensorflowDetections[idx].category.label = "left front wheel"
                        } else {
                            resultState.tensorflowDetections[idx].category.label = "right front wheel"
                        }
                    } else if (partclosestowheelleftlabel == "rear bumper cover") {
                        if (partclosesttowheelleft.boundingBox.left < wheelleft) {
                            resultState.tensorflowDetections[idx].category.label = "right rear wheel"
                        } else {
                            resultState.tensorflowDetections[idx].category.label = "left rear wheel"
                        }
                    }
                }
            }
        }
        //Remove duplicates and save the one with the highest score
        resultState.tensorflowDetections = resultState.tensorflowDetections.groupBy{
            it.category.label
        }.mapValues { (_, sameNameDetections) ->
            sameNameDetections.maxBy {it.category.score}
        }.values.toMutableList()

        val scaleFactor = max(boxsize.width / resultState.tensorflowImageWidth, boxsize.height / resultState.tensorflowImageHeight)
        resultState.tensorflowDetections.forEachIndexed { index, detection ->
            val boundingBox = detection.boundingBox

            //Once returned, the bounding boxes and coordinates need to be scaled back up to
            //be correctly displayed on screen
            var top = boundingBox.top * scaleFactor
            var left = boundingBox.left * scaleFactor
            var bottom = boundingBox.bottom * scaleFactor
            var right = boundingBox.right * scaleFactor

            var centerX = (left + right) / 2
            var centerY = (top + bottom) / 2
            top = centerY - 25f
            bottom = centerY + 25f
            left = centerX - 25f
            right = centerX + 25f

            resultState.tensorflowDetections[index].boundingBox.set(left, top, right, bottom)
        }

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        avgInference.add(inferenceTime)
        val avg = avgInference.average()
        if(avgInference.size > 500) {
            avgInference.clear()
        }
        viewModelScope.launch(Dispatchers.Main){
            _detectionState.value = _detectionState.value.copy(
                inferenceTimeCurr = inferenceTime,
                inferenceTimeAvg = avg,
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