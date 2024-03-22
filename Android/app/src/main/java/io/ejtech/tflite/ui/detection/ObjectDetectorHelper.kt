/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.ejtech.tflite.ui.detection

import android.content.Context
import android.graphics.Bitmap
import androidx.compose.runtime.mutableStateOf
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.gpu.support.TfLiteGpu
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.gms.vision.TfLiteVision
import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector

class ObjectDetectorHelper(
  var threshold: Float = 0.8f,
  var numThreads: Int = 4,
  var maxResults: Int = 10,
  var modelName: String = "detect_coin.tflite",
  val context: Context
) {
    private var _detectorState = mutableStateOf(DetectionState())
    private var objectDetectorHelper: ObjectDetector? = null

     fun initialize() = callbackFlow<Resource<DetectionState>> {
         if(objectDetectorHelper == null){
             TfLiteGpu.isGpuDelegateAvailable(context).onSuccessTask { gpuAvailable: Boolean ->
                 val optionsBuilder = TfLiteInitializationOptions.builder()
                 TfLiteVision.initialize(context, optionsBuilder.build())
             }.addOnSuccessListener {
                 val optionsBuilder = ObjectDetector.ObjectDetectorOptions.builder()
                     .setScoreThreshold(threshold)
                     .setMaxResults(maxResults)
                 val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)
                 optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

                 try {
                     objectDetectorHelper = ObjectDetector.createFromFileAndOptions(context, modelName, optionsBuilder.build())
                     _detectorState.value = _detectorState.value.copy(tensorflowEnabled = true)
                     launch{
                         send(Resource.Success(_detectorState.value))
                     }
                 } catch (e: Exception) {
                     _detectorState.value = _detectorState.value.copy(tensorflowEnabled = false)
                     launch{
                         send(Resource.Success(_detectorState.value))
                     }
                 }
             }.addOnFailureListener{
                 _detectorState.value = _detectorState.value.copy(tensorflowEnabled = false)
                 launch{
                     send(Resource.Success(_detectorState.value))
                 }
             }
         }
         else{
             _detectorState.value = _detectorState.value.copy(tensorflowEnabled = true)
             launch{
                 send(Resource.Success(_detectorState.value))
             }
         }
         awaitClose {  }
    }

    fun detect(image: Bitmap, imageRotation: Int): DetectionState {
        val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
        val results = objectDetectorHelper?.detect(tensorImage)

        results?.let{
            _detectorState.value = _detectorState.value.copy(
                tensorflowDetections = results,
                tensorflowImageHeight = tensorImage.height,
                tensorflowImageWidth = tensorImage.width
            )
        }
        return _detectorState.value
    }

    fun destroy(): DetectionState {
        _detectorState.value = DetectionState()
        objectDetectorHelper?.close()
        objectDetectorHelper = null
        return _detectorState.value
    }
}
