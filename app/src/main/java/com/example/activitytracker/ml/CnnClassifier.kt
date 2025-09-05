package com.example.activitytracker.ml

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.nio.FloatBuffer

class CnnClassifier(private val context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    @Volatile private var session: OrtSession? = null

    private fun ensureSession(): OrtSession {
        val cached = session
        if (cached != null) return cached

        // Load model bytes from assets: app/src/main/assets/cnn.onnx
        val bytes = context.assets.open(MODEL_ASSET_PATH).use { it.readBytes() }
        val s = env.createSession(bytes)
        session = s
        return s
    }

    /**
     * Run inference.
     * Supports two inputs:
     *  - Raw flattened: [1, N*3] (x1,y1,z1,x2,...) for 1D CNN exports
     *  - Feature vector: [1, 43] for classical/MLP exports
     * Chooses based on input tensor static second dimension if available (43 => features),
     * otherwise defaults to raw flattening.
     * Returns Pair(classIndex, probabilities[4])
     */
    fun classify(window: List<Triple<Float, Float, Float>>, samplingHz: Double): Pair<Int, FloatArray> {
        if (window.isEmpty()) return 0 to floatArrayOf(1f, 0f, 0f, 0f)
        val sess = ensureSession()

        val win = window.size
        val inputName = sess.inputNames.find { it.equals("input", ignoreCase = true) } ?: sess.inputNames.first()

        // Detect expected length if statically known
        val declaredLen: Int? = try {
            val ni = sess.inputInfo[inputName]
            val ti = (ni?.info as? TensorInfo)
            val shape = ti?.shape
            if (shape != null && shape.size >= 2) {
                val dim = shape[1]
                if (dim > 0) dim.toInt() else null
            } else null
        } catch (_: Throwable) { null }

        // Prepare both representations
        val feats = FeatureExtractor.extract(window, samplingHz, numBins = 10)
        val raw = FloatArray(win * 3).also { arr ->
            var idx = 0
            window.forEach { (x, y, z) -> arr[idx++] = x; arr[idx++] = y; arr[idx++] = z }
        }

        fun runOnce(data: FloatArray): Pair<Int, FloatArray> {
            val shape = longArrayOf(1, data.size.toLong())
            OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape).use { tensor ->
                val results = sess.run(mapOf(inputName to tensor))
                results.use {
                    val value = it[0].value as Array<FloatArray>
                    val out = value[0]
                    // If output does not appear normalized, apply softmax for stability
                    val sum = out.sum()
                    val probs = if (sum.isFinite() && sum > 0.0f && sum <= 1.0001f) out else {
                        val maxLogit = out.maxOrNull() ?: 0f
                        val exps = FloatArray(out.size) { i -> kotlin.math.exp((out[i] - maxLogit).toDouble()).toFloat() }
                        val denom = exps.sum().coerceAtLeast(1e-6f)
                        FloatArray(out.size) { i -> exps[i] / denom }
                    }
                    var argmax = 0
                    var best = probs[0]
                    for (i in 1 until probs.size) if (probs[i] > best) { best = probs[i]; argmax = i }
                    return argmax to probs
                }
            }
        }

        // Choose preferred representation based on declared input length, with fallback if it fails
        val preferFeatures = (declaredLen == 43)
        return try {
            if (preferFeatures) runOnce(feats) else runOnce(raw)
        } catch (_: Throwable) {
            // Fallback to the other representation
            try {
                if (preferFeatures) runOnce(raw) else runOnce(feats)
            } catch (e: Throwable) {
                // As a last resort, return class 0
                0 to floatArrayOf(1f, 0f, 0f, 0f)
            }
        }
    }

    companion object {
        const val MODEL_ASSET_PATH = "cnn.onnx"
    }
}
