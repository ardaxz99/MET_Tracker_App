package com.example.activitytracker.ml

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

object FeatureExtractor {
    /**
     * Extract 43 features from a window of (x,y,z) samples.
     * - Means, STDs, mean absolute deviation per axis (9)
     * - Average resultant acceleration (1)
     * - Average time between simple local peaks per axis in ms (3)
     * - Histogram (numBins per axis, normalized) => 30 (default 10 bins/axis)
     */
    fun extract(window: List<Triple<Float, Float, Float>>, samplingHz: Double, numBins: Int = 10): FloatArray {
        val n = window.size
        require(n > 0) { "Window must be non-empty" }

        val xs = FloatArray(n)
        val ys = FloatArray(n)
        val zs = FloatArray(n)
        for (i in 0 until n) {
            val (x, y, z) = window[i]
            xs[i] = x; ys[i] = y; zs[i] = z
        }

        val feats = ArrayList<Float>(43)

        fun mean(a: FloatArray): Double {
            var s = 0.0
            for (v in a) s += v
            return s / a.size
        }
        fun std(a: FloatArray, mu: Double = mean(a)): Double {
            var s = 0.0
            for (v in a) { val d = v - mu; s += d * d }
            return sqrt(s / a.size)
        }
        fun mad(a: FloatArray, mu: Double = mean(a)): Double {
            var s = 0.0
            for (v in a) s += abs(v - mu).toDouble()
            return s / a.size
        }

        // Per-axis stats
        val mux = mean(xs); val muy = mean(ys); val muz = mean(zs)
        val sdx = std(xs, mux); val sdy = std(ys, muy); val sdz = std(zs, muz)
        val madx = mad(xs, mux); val mady = mad(ys, muy); val madz = mad(zs, muz)
        feats.add(mux.toFloat()); feats.add(sdx.toFloat()); feats.add(madx.toFloat())
        feats.add(muy.toFloat()); feats.add(sdy.toFloat()); feats.add(mady.toFloat())
        feats.add(muz.toFloat()); feats.add(sdz.toFloat()); feats.add(madz.toFloat())

        // Average resultant acceleration
        var sumRes = 0.0
        for (i in 0 until n) {
            val x = xs[i].toDouble(); val y = ys[i].toDouble(); val z = zs[i].toDouble()
            sumRes += sqrt(x * x + y * y + z * z)
        }
        feats.add((sumRes / n).toFloat())

        // Time between simple local peaks (ms) per axis
        fun avgPeakDistanceMs(a: FloatArray): Float {
            if (n < 3 || samplingHz <= 0.0) return 0f
            val peaks = ArrayList<Int>()
            for (i in 1 until n - 1) {
                val prev = a[i - 1]
                val cur = a[i]
                val next = a[i + 1]
                if (cur > prev && cur > next) peaks.add(i)
            }
            if (peaks.size < 2) return 0f
            var sum = 0.0
            for (i in 1 until peaks.size) sum += (peaks[i] - peaks[i - 1]).toDouble()
            val avgDist = sum / (peaks.size - 1)
            return ((avgDist / samplingHz) * 1000.0).toFloat()
        }
        feats.add(avgPeakDistanceMs(xs))
        feats.add(avgPeakDistanceMs(ys))
        feats.add(avgPeakDistanceMs(zs))

        // Histograms per axis (normalized)
        fun hist(a: FloatArray, bins: Int): FloatArray {
            var minV = a[0].toDouble()
            var maxV = a[0].toDouble()
            for (v in a) { if (v < minV) minV = v.toDouble(); if (v > maxV) maxV = v.toDouble() }
            if (maxV <= minV) maxV = minV + 1e-6
            val counts = FloatArray(bins)
            val width = (maxV - minV) / bins
            for (v in a) {
                var idx = floor(((v - minV) / width)).toInt()
                if (idx < 0) idx = 0
                if (idx >= bins) idx = bins - 1
                counts[idx] += 1f
            }
            val invN = 1f / a.size.toFloat()
            for (i in 0 until bins) counts[i] *= invN
            return counts
        }
        hist(xs, numBins).forEach { feats.add(it) }
        hist(ys, numBins).forEach { feats.add(it) }
        hist(zs, numBins).forEach { feats.add(it) }

        // 43 elements total
        return feats.toFloatArray()
    }
}

