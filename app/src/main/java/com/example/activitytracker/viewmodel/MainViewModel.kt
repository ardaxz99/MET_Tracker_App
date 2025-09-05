package com.example.activitytracker.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.activitytracker.data.repo.ActivityRepository
import com.example.activitytracker.domain.ActivityState
import com.example.activitytracker.domain.displayName
import com.example.activitytracker.prefs.SettingsStore
import com.example.activitytracker.sensors.SensorService
import com.example.activitytracker.ml.CnnClassifier
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.roundToInt

class MainViewModel(
    private val repo: ActivityRepository,
    private val sensor: SensorService,
    private val settings: SettingsStore,
    private val classifier: CnnClassifier? = null
) : ViewModel() {

    val imuEnabled: StateFlow<Boolean> = settings.showImuFlow
        .stateIn(viewModelScope, kotlinx.coroutines.flow.SharingStarted.Eagerly, false)

    val liveAccel = sensor.liveAccel.stateIn(
        viewModelScope,
        kotlinx.coroutines.flow.SharingStarted.Eagerly,
        Triple(0f, 0f, 0f)
    )

    val samplingRateHz = sensor.samplingRateHz
        .stateIn(viewModelScope, kotlinx.coroutines.flow.SharingStarted.Eagerly, 20.0)

    // Real-time computed target number of samples per window
    private val _windowSamples = MutableStateFlow(0)
    val windowSamples: StateFlow<Int> = _windowSamples.asStateFlow()

    // Real-time current buffer fill (samples currently in window)
    private val _windowFillSamples = MutableStateFlow(0)
    val windowFillSamples: StateFlow<Int> = _windowFillSamples.asStateFlow()

    // Latest CNN probability distribution [Sedentary, Light, Moderate, Vigorous]
    private val _cnnProbs = MutableStateFlow(listOf(0.0, 0.0, 0.0, 0.0))
    val cnnProbs: StateFlow<List<Double>> = _cnnProbs.asStateFlow()

    val windowSeconds = settings.windowSecondsFlow
        .stateIn(viewModelScope, kotlinx.coroutines.flow.SharingStarted.Eagerly, 10.0)
    val strideSeconds = settings.strideSecondsFlow
        .stateIn(viewModelScope, kotlinx.coroutines.flow.SharingStarted.Eagerly, 5.0)

    val themeDark = settings.themeDarkFlow
        .stateIn(viewModelScope, kotlinx.coroutines.flow.SharingStarted.Eagerly, false)

    val ageYears = settings.ageYearsFlow
        .stateIn(viewModelScope, kotlinx.coroutines.flow.SharingStarted.Eagerly, 0)
    val weightKg = settings.weightKgFlow
        .stateIn(viewModelScope, kotlinx.coroutines.flow.SharingStarted.Eagerly, 0.0)
    val heightCm = settings.heightCmFlow
        .stateIn(viewModelScope, kotlinx.coroutines.flow.SharingStarted.Eagerly, 0.0)
    val gender = settings.genderFlow
        .stateIn(viewModelScope, kotlinx.coroutines.flow.SharingStarted.Eagerly, "unspecified")

    private val _activityState = MutableStateFlow(ActivityState.Sedentary)
    val activityState: StateFlow<ActivityState> = _activityState.asStateFlow()

    init {
        // Build classification pipeline: maintain window and classify with ONNX CNN
        viewModelScope.launch {
            var buffer = ArrayDeque<Double>()
            var bufferXYZ = ArrayDeque<Triple<Float, Float, Float>>()
            var strideCount = 0
            var currentWindowSize = 0
            var currentStride = 1
            var lastHzInt = 0
            var lastWSec = -1.0
            var lastSSec = -1.0
            var lastRecordedSec: Long = 0L

            // Gravity low-pass filter state (per-axis). Null until first sample initializes it.
            var gX: Double? = null
            var gY: Double? = null
            var gZ: Double? = null
            val fc = 0.4 // Hz, cutoff for gravity LPF (0.3–0.5 typical)
            combine(sensor.liveAccel, samplingRateHz, windowSeconds, strideSeconds) { triple, hz, wSec, sSec ->
                Quadruple(triple, hz, wSec, sSec)
            }.collect { (triple, hz, wSec, sSec) ->
                val hzInt = hz.roundToInt().coerceIn(1, 200)
                val newWindowSize = (hzInt * wSec).roundToInt().coerceIn(5, 2000)
                val newStride = (hzInt * sSec).roundToInt().coerceIn(1, 2000)

                // Publish current window size (samples) for UI
                _windowSamples.value = newWindowSize

                val windowChangedSignificantly = currentWindowSize == 0 ||
                        kotlin.math.abs(newWindowSize - currentWindowSize) >= kotlin.math.max(3, currentWindowSize / 10)
                val settingsChanged = (hzInt != lastHzInt) || (wSec != lastWSec) || (sSec != lastSSec)
                if (settingsChanged && windowChangedSignificantly) {
                    // Recreate/trim buffers when target size changes notably
                    // Magnitude buffer
                    val newBuf = ArrayDeque<Double>()
                    while (buffer.size > newWindowSize) buffer.removeFirst()
                    buffer.forEach { newBuf.addLast(it) }
                    buffer = newBuf
                    // XYZ buffer
                    val newBufXYZ = ArrayDeque<Triple<Float, Float, Float>>()
                    while (bufferXYZ.size > newWindowSize) bufferXYZ.removeFirst()
                    bufferXYZ.forEach { newBufXYZ.addLast(it) }
                    bufferXYZ = newBufXYZ

                    currentWindowSize = newWindowSize
                    strideCount = 0
                    _windowFillSamples.value = buffer.size
                }
                currentStride = newStride
                lastHzInt = hzInt
                lastWSec = wSec
                lastSSec = sSec

                // Magnitude in g (normalize by gravity) from raw accel
                val mag = sqrt(
                    (triple.first.toDouble().pow(2) +
                            triple.second.toDouble().pow(2) +
                            triple.third.toDouble().pow(2))
                ) / 9.80665

                if (currentWindowSize <= 0) currentWindowSize = 5
                // Keep buffers bounded even if window shrinks between ticks
                while (buffer.size >= currentWindowSize && buffer.isNotEmpty()) buffer.removeFirst()
                buffer.addLast(mag)
                while (bufferXYZ.size >= currentWindowSize && bufferXYZ.isNotEmpty()) bufferXYZ.removeFirst()
                // Feed CNN with raw accelerometer values
                bufferXYZ.addLast(triple)
                _windowFillSamples.value = buffer.size
                strideCount++
                if (buffer.size >= currentWindowSize && strideCount % currentStride == 0) {
                    val result = runCnn(bufferXYZ, hz)
                    if (result != null) {
                        val (state, probs) = result
                        _activityState.value = state
                        _cnnProbs.value = probs
                        // Record per-second state once per wall-clock second
                        val nowSec = System.currentTimeMillis() / 1000L
                        if (nowSec != lastRecordedSec) {
                            lastRecordedSec = nowSec
                            val label = state.displayName
                            viewModelScope.launch { repo.recordSecond(nowSec, label) }
                        }
                    }
                }
            }
        }
    }

    fun setImuEnabled(show: Boolean) {
        viewModelScope.launch { settings.setShowImu(show) }
    }

    fun setThemeDark(dark: Boolean) {
        viewModelScope.launch { settings.setThemeDark(dark) }
    }

    fun saveUserProfile(ageYears: Int, weightKg: Double, heightCm: Double, gender: String) {
        viewModelScope.launch { settings.setUserProfile(ageYears, weightKg, heightCm, gender) }
    }

    val history = repo.recentSlices()

    fun resetAllData() {
        viewModelScope.launch { repo.clearAll() }
    }

    // Expose repo aggregation for Day view
    fun repoDayHourCounts(dayStartSec: Long) = repo.dayHourCounts(dayStartSec)
    fun repoWeekCounts(weekStartSec: Long) = repo.weekDayCounts(weekStartSec)
    fun repoMonthCounts(monthStartSec: Long, daysInMonth: Int) = repo.monthDayCounts(monthStartSec, daysInMonth)
    fun repoYearCounts(yearStartSec: Long) = repo.yearMonthCounts(yearStartSec)

    private fun stdDev(values: Collection<Double>): Double {
        val mean = values.average()
        val variance = values.fold(0.0) { acc, d -> acc + (d - mean) * (d - mean) } / values.size
        return sqrt(variance)
    }

    private fun runCnn(window: Collection<Triple<Float, Float, Float>>, samplingHz: Double): Pair<ActivityState, List<Double>>? {
        val clf = classifier ?: return null
        if (window.isEmpty()) return null
        return try {
            val (idx, probs) = clf.classify(window.toList(), samplingHz)
            val state = when (idx) {
                0 -> ActivityState.Sedentary
                1 -> ActivityState.Small
                2 -> ActivityState.Medium
                3 -> ActivityState.Vigorous
                else -> ActivityState.Sedentary
            }
            state to probs.map { it.toDouble() }
        } catch (_: Throwable) {
            null
        }
    }
}

// Lightweight tuple type for combine result
private data class Quadruple<A,B,C,D>(
    val a: A,
    val b: B,
    val c: C,
    val d: D
)
