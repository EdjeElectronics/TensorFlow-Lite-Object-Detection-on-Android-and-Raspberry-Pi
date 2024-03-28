package io.ejtech.tflite

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel
import dagger.hilt.android.AndroidEntryPoint
import io.ejtech.tflite.ui.detection.DetectionScreen
import io.ejtech.tflite.ui.detection.DetectionViewModel
import io.ejtech.tflite.ui.theme.MyApplicationTheme

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        keepScreenOn()
        setContent {
            MyApplicationTheme {
                //How DetectionScreen is displayed
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val detectionViewModel = hiltViewModel<DetectionViewModel>()
                    DetectionScreen(
                        detectionViewModel = detectionViewModel,
                        detectionState = detectionViewModel.detectionState.value
                    )
                }
            }
        }
    }

    /**
     * Prevents screen from sleeping
     */
    private fun keepScreenOn() {
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }
}