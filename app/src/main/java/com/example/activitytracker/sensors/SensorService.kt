package com.example.activitytracker.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.sqrt

data class AccelSample(val x: Float, val y: Float, val z: Float, val t: Long) {
    fun magnitude(): Double = sqrt((x * x + y * y + z * z).toDouble())
}

class SensorService(context: Context) {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    // Request ~20 Hz; actual may vary per device. We'll resample to a stable target.
    private val samplingPeriodUs = 50_000 // ~20 Hz request

    // Single source flow of accelerometer samples (raw device stream)
    val samples: Flow<AccelSample> = callbackFlow {
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                trySend(AccelSample(event.values[0], event.values[1], event.values[2], event.timestamp))
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }
        sensorManager.registerListener(listener, accel, samplingPeriodUs)
        awaitClose { sensorManager.unregisterListener(listener) }
    }

    // Resample to a stable target rate (~20 Hz) using sample-and-hold so downstream logic sees a steady rate
    private val targetHz = 20.0
    private val periodMs = (1000.0 / targetHz).toLong().coerceAtLeast(1L)

    private val samplesResampled: Flow<AccelSample> = kotlinx.coroutines.flow.flow {
        var last: AccelSample? = null
        // Collect raw samples into `last` while emitting at fixed period using sample-and-hold
        coroutineScope {
            val job = launch {
                samples.collect { s -> last = s }
            }
            // Wait until the first sensor sample arrives
            while (last == null) { delay(5) }
            while (true) {
                val s = last!!
                // Use a monotonic host timestamp for the resampled emission
                emit(AccelSample(s.x, s.y, s.z, System.nanoTime()))
                delay(periodMs)
            }
            job.cancel()
        }
    }

    // Convenience triple for UI bindings (resampled)
    val liveAccel: Flow<Triple<Float, Float, Float>> = samplesResampled.map { Triple(it.x, it.y, it.z) }

    // Measured sampling rate in Hz (EMA smoothed) based on resampled stream
    val samplingRateHz: Flow<Double> = kotlinx.coroutines.flow.flow {
        var lastTs: Long? = null
        var ema: Double? = null
        samplesResampled.collect { s ->
            val prev = lastTs
            lastTs = s.t
            if (prev != null) {
                val hz = 1e9 / (s.t - prev).coerceAtLeast(1L).toDouble()
                ema = if (ema == null) hz else (0.9 * (ema!!) + 0.1 * hz)
                emit(ema!!)
            }
        }
    }
}
