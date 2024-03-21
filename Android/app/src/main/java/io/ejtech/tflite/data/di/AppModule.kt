package io.ejtech.tflite.data.di

import android.app.Application
import android.content.Context
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import io.ejtech.tflite.ui.detection.ObjectDetectorHelper
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {
    @Provides
    @Singleton
    fun provideContext(application: Application): Context {
        return application.applicationContext
    }

    @Provides
    @Singleton
    fun provideObjectDetectorHelper(context: Context): ObjectDetectorHelper {
        return ObjectDetectorHelper(context = context)
    }
}