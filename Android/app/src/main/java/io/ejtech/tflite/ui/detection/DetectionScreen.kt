package io.ejtech.tflite.ui.detection

import android.Manifest
import android.app.Activity
import android.content.pm.ActivityInfo
import android.graphics.Bitmap
import android.graphics.Paint.Align
import android.view.OrientationEventListener
import android.view.Surface
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.absoluteOffset
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.size
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
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.toSize
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max

@Composable
fun DetectionScreen(
    detectionViewModel: DetectionViewModel,
    detectionState: DetectionState
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val activity = (LocalContext.current as Activity)
    val configuration = LocalConfiguration.current
    activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE

    var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    // Shutdown executor when screen is disposed of
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_DESTROY) {
                cameraExecutor.shutdown()
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    var bitmapBuffer: Bitmap? = null
    var imageCapture: ImageCapture = remember {
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
    }

    val degreeRotationState = remember { mutableStateOf(0f) }
    val rotationState by animateFloatAsState(
        targetValue = degreeRotationState.value, label = ""
    )

    var orientationEventListener = remember { mutableStateOf(object: OrientationEventListener(context){
        override fun onOrientationChanged(orientation: Int) {
            if ((orientation < 35 || orientation > 325)) { // PORTRAIT
                imageCapture.targetRotation = Surface.ROTATION_0
                degreeRotationState.value = 270f
            } else if (orientation in 146..214) { // REVERSE PORTRAIT
                imageCapture.targetRotation = Surface.ROTATION_180
                degreeRotationState.value = 90f
            } else if (orientation in 56..124) { // REVERSE LANDSCAPE
                imageCapture.targetRotation = Surface.ROTATION_270
                degreeRotationState.value = 180f
            } else if (orientation in 236..304) { //LANDSCAPE
                imageCapture.targetRotation = Surface.ROTATION_90
                degreeRotationState.value = 0f
            }
        }
    }) }
    //Hack to prevent crash from too many listeners being enabled when rotating the phone too many times
    //Disable before each enable
    var orientationListenerEnabled = remember{ mutableStateOf(false) }
    if(!orientationListenerEnabled.value) {
        orientationListenerEnabled.value = true
        orientationEventListener.value.enable()
    }

    val camera = remember { mutableStateOf<Camera?>(null) }
    val cameraProviderFuture = remember {
        ProcessCameraProvider.getInstance(context)
    }

    val permissionsLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions(),
        onResult = { granted ->
            //hasWriteExternalStoragePermission = granted
        }
    )
    LaunchedEffect(key1 = true) {
        permissionsLauncher.launch(
            arrayOf(
                Manifest.permission.CAMERA
            )
        )
    }

    var boxsize by remember { mutableStateOf(IntSize.Zero)}
    Box(
        modifier = Modifier
            .fillMaxSize()
    ){
        AndroidView(
            modifier = Modifier
                .fillMaxSize()
                .onGloballyPositioned { coordinates ->
                    boxsize = coordinates.size
                },
            factory = { context ->
                PreviewView(context).also{
                    val preview = Preview.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .build()
                    val selector = CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build()
                    preview.setSurfaceProvider(it.surfaceProvider)

                    var imageAnalyzer: ImageAnalysis = ImageAnalysis.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()
                        .also{
                            it.setAnalyzer(cameraExecutor) { image ->
                                if (bitmapBuffer == null) {
                                    // The image rotation and RGB image buffer are initialized only once
                                    // the analyzer has started running
                                    bitmapBuffer = Bitmap.createBitmap(
                                        image.width,
                                        image.height,
                                        Bitmap.Config.ARGB_8888
                                    )
                                }
                                if(bitmapBuffer != null){
                                    detectionViewModel.detectObjects(image, bitmapBuffer!!)
                                }
                            }
                        }

                    try{
                        //Need to unbind when we return from preview or it will rebind the camera and screen will remain black
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

        if(detectionState.tensorflowEnabled && detectionState.tensorflowDetections.isNotEmpty()){
            val imageWidth = detectionState.tensorflowImageWidth
            val imageHeight = detectionState.tensorflowImageHeight

            for (detection in detectionState.tensorflowDetections){
                val boundingBox = detection.boundingBox
                val widthScale = 2168f / imageWidth
                val heightScale = 1008f / imageHeight

                val top = boundingBox.top * widthScale
                val bottom = boundingBox.bottom * widthScale
                val left = boundingBox.left * widthScale
                val right = boundingBox.right * widthScale

                var label = detection.categories[0].label

                Canvas(
                    modifier = Modifier
                        .fillMaxSize()
                ) {
                    val rect = Rect(left, top, right, bottom)
                    drawRect(
                        topLeft = Offset(left, top),
                        color = Color.Green,
                        style = Stroke(width = 8f),
                        size = rect.size)
                }

                /*
                Column(
                ){
                    Text(
                        modifier = Modifier
                            .background(Color.Black),
                        text = label + (boundingBox.width().dp+8.dp).toString() + (boundingBox.height().dp+8.dp).toString(),
                        fontSize = 16.sp,
                        textAlign = TextAlign.Center,
                        fontWeight = FontWeight.Bold,
                        color = Color.White,
                    )
                    Box(
                        modifier = Modifier
                            //.absoluteOffset(left.dp-4.dp, top.dp-4.dp)
                            .offset(left.dp-4.dp, top.dp-4.dp)
                            .background(Color.Transparent)
                            .border(2.dp, Color.Black)
                            .size(boundingBox.width().dp+8.dp, boundingBox.height().dp+8.dp)
                    ) {}
                }*/
            }
        }
    }

}