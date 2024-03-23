package io.ejtech.tflite.ui.detection

import android.Manifest
import android.app.Activity
import android.content.pm.ActivityInfo
import android.graphics.Bitmap
import android.graphics.Paint
import android.graphics.Paint.Align
import android.graphics.Rect
import android.graphics.RectF
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
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign

import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.toSize
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.graphics.toRect
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
    activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE

    var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
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
    val camera = remember { mutableStateOf<Camera?>(null) }
    val cameraProviderFuture = remember {
        ProcessCameraProvider.getInstance(context)
    }

    val permissionsLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions(),
        onResult = { granted ->

        }
    )
    LaunchedEffect(key1 = true) {
        permissionsLauncher.launch(
            arrayOf(
                Manifest.permission.CAMERA
            )
        )
    }

    val textMeasurer = rememberTextMeasurer()

    Box(
        modifier = Modifier
            .fillMaxSize()
    ){
        AndroidView(
            modifier = Modifier
                .fillMaxSize(),
            factory = { context ->
                PreviewView(context).also{
                    it.scaleType = PreviewView.ScaleType.FILL_START
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

        if(detectionState.tensorflowEnabled && detectionState.tensorflowDetections.isNotEmpty()) {
            Canvas(
                modifier = Modifier
                    .fillMaxSize()
            ) {
                val scaleFactor = max(size.width / detectionState.tensorflowImageWidth, size.height / detectionState.tensorflowImageHeight)
                for (detection in detectionState.tensorflowDetections) {
                    val boundingBox = detection.boundingBox

                    val top = boundingBox.top * scaleFactor
                    val left = boundingBox.left * scaleFactor
                    val bottom = boundingBox.bottom * scaleFactor
                    val right = boundingBox.right * scaleFactor
                    val scaledRect = RectF(left, top, right, bottom)

                    var label = detection.categories[0].label

                    drawRect(
                        topLeft = Offset(left, top),
                        color = Color.Green,
                        style = Stroke(width = 3.dp.toPx()),
                        size = Size(scaledRect.width(), scaledRect.height())
                    )

                    val textBackgroundPaint = Paint()
                    textBackgroundPaint.textSize = 14.sp.toPx()
                    textBackgroundPaint.getTextBounds(label, 0, label.length, scaledRect.toRect())
                    val textWidth = scaledRect.width()
                    val textHeight = scaledRect.height()

                    drawRect(
                        topLeft = Offset(x = left, y = top - 60),
                        color = Color.Black,
                        size = Size(
                            width = textWidth,
                            height = 50F
                        ),
                    )

                    drawText(
                        textMeasurer = textMeasurer,
                        text = label,
                        topLeft = Offset(x = left, y = top - 75),
                        style = TextStyle(
                            color = Color.White,
                            fontSize = 20.sp,
                            fontWeight = FontWeight.ExtraBold
                        ),
                        size = Size(
                            width = textWidth + 30f,
                            height = textHeight + 30f
                        )
                    )
                }
            }
        }
    }
}