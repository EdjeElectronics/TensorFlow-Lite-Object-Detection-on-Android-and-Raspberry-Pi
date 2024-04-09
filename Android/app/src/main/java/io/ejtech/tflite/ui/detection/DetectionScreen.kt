package io.ejtech.tflite.ui.detection

import android.Manifest
import android.app.Activity
import android.content.pm.ActivityInfo
import android.graphics.Bitmap
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer

import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.toSize
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.graphics.toRect
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.navigation.NavGraph
import com.google.android.gms.tflite.gpu.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max

/**
 * Main screen that displays the camera view and visible detections
 *
 * @param detectionViewModel
 *      Receives images from the camera feed and returns detections
 * @param detectionState
 *      Holds the detections returned from detectionViewModel
 */
@Composable
fun DetectionScreen(
    detectionViewModel: DetectionViewModel,
    detectionState: DetectionState
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val activity = (LocalContext.current as Activity)
    // Set the screen to remain in Landscape mode no matter how the device is held
    activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE

    // Runs camera on a separate thread and observes this screen to end the thread when the screen is destroyed
    var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_DESTROY) {
                cameraExecutor.shutdown()
                detectionViewModel.destroy()
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    // Camera state
    val camera = remember { mutableStateOf<Camera?>(null) }
    val cameraProviderFuture = remember {
        ProcessCameraProvider.getInstance(context)
    }

    //Requests permission from user to gain access to the camera
    val permissionsLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions(),
        onResult = { granted -> }
    )
    LaunchedEffect(key1 = true) {
        permissionsLauncher.launch(
            arrayOf(
                Manifest.permission.CAMERA
            )
        )
    }

    //Listed for a one-time initialization event of Tensorflow
    LaunchedEffect(key1 = context) {
        detectionViewModel.tensorflowInitializationEvent.collect { event ->
            when (event) {
                is Resource.Success -> {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(context, event.data, Toast.LENGTH_SHORT).show()
                    }
                }
                is Resource.Loading -> {

                }
                is Resource.Error -> {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(context, event.message, Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }

    var boxsize by remember { mutableStateOf(Size.Zero)}
    //Used to size the text label on detections
    val textMeasurer = rememberTextMeasurer()
    // Stores the image for each camera frame
    var imageBitmap: Bitmap? = null
    Box(
        modifier = Modifier
            .fillMaxSize()
            .onGloballyPositioned { coordinates ->
                boxsize = coordinates.size.toSize()
            }
    ){
        AndroidView(
            modifier = Modifier
                .fillMaxSize(),
            factory = { context ->
                //PreviewView is the camera preview
                PreviewView(context).also{
                    //Fill the camera view to the entire screen
                    it.scaleType = PreviewView.ScaleType.FILL_START
                    //Ratio that best matches our model image format
                    val preview = Preview.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .build()
                    //Use the rear camera
                    val selector = CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build()
                    preview.setSurfaceProvider(it.surfaceProvider)

                    //Passes each camera frame to the viewmodel to detect objects
                    var imageAnalyzer: ImageAnalysis = ImageAnalysis.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()
                        .also{
                            it.setAnalyzer(cameraExecutor) { image ->
                                if (imageBitmap == null) {
                                    imageBitmap = Bitmap.createBitmap(
                                        image.width,
                                        image.height,
                                        Bitmap.Config.ARGB_8888
                                    )
                                }
                                detectionViewModel.detectObjects(image, imageBitmap!!, boxsize)
                            }
                        }

                    //Assigns our imageAnalyzer to the camera
                    try{
                        cameraProviderFuture.get().unbindAll()
                        camera.value = cameraProviderFuture.get().bindToLifecycle(
                            lifecycleOwner,
                            selector,
                            preview,
                            imageAnalyzer
                        )
                    } catch(e: Exception){
                        e.printStackTrace()
                    }
                }
            }
        )
        Text(text = detectionState.inferenceTime.toString(), modifier = Modifier.align(Alignment.BottomCenter))

        //Where detections are drawn on screen if the model successfully load and the screen has detections
        if(detectionState.tensorflowEnabled && detectionState.tensorflowDetections.isNotEmpty()) {
            Canvas(
                modifier = Modifier
                    .fillMaxSize()
            ) {
                //Images are resized before being passed to the ObjectDetector
                for (detection in detectionState.tensorflowDetections) {
                    val boundingBox = detection.boundingBox
                    val label = detection.category.label

                    //Draws the bounding box
                    drawRect(
                        topLeft = Offset(boundingBox.left, boundingBox.top),
                        color = Color.Green,
                        style = Stroke(width = 3.dp.toPx()),
                        size = Size(boundingBox.width(), boundingBox.height())
                    )

                    val textBounds = Rect()
                    val textPaint = Paint().apply {
                        textSize = 14.sp.toPx()
                        color = ContextCompat.getColor(context, com.example.myapplication.R.color.white)
                    }
                    textPaint.getTextBounds(label, 0, label.length, textBounds)

                    val backgroundRect = androidx.compose.ui.geometry.Rect(
                        0f,
                        0f,
                        textBounds.width().toFloat(),
                        textBounds.height().toFloat()
                    )
                    drawRect(
                        color = Color.Black,
                        topLeft = Offset(x = boundingBox.left, y = boundingBox.top - 50),
                        size = backgroundRect.size
                    )
                    drawContext.canvas.nativeCanvas.drawText(
                        label,
                        boundingBox.left,
                        boundingBox.top - 60 + textBounds.height(),
                        textPaint
                    )
                }
            }
        }
    }
}