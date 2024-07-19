package dev.flower.flower_tflite

import android.util.Log
import dev.flower.flower_tflite.helpers.assertIntsEqual
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.lang.Integer.min
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.util.concurrent.locks.ReentrantLock
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.withLock
import kotlin.concurrent.write

class FlowerClient<X : Any, Y : Any>(
    tfliteFileBuffer: MappedByteBuffer,
    val layersSizes: IntArray,
    val spec: SampleSpec<X, Y>,
) : AutoCloseable {
    val interpreter = Interpreter(tfliteFileBuffer)
    val interpreterLock = ReentrantLock()
    val trainingSamples = mutableListOf<Sample<X, Y>>()
    val testSamples = mutableListOf<Sample<X, Y>>()
    val trainSampleLock = ReentrantReadWriteLock()
    val testSampleLock = ReentrantReadWriteLock()
    private var fitJob: Job? = null

    fun addSample(
        bottleneck: X, label: Y, isTraining: Boolean
    ) {
        val samples = if (isTraining) trainingSamples else testSamples
        val lock = if (isTraining) trainSampleLock else testSampleLock
        lock.write {
            samples.add(Sample(bottleneck, label))
        }
    }

    fun getParameters(): Array<ByteBuffer> {
        val inputs: Map<String, Any> = FakeNonEmptyMap()
        val outputs = emptyParameterMap()
        runSignatureLocked(inputs, outputs, "parameters")
        return parametersFromMap(outputs)
    }

    fun updateParameters(parameters: Array<ByteBuffer>): Array<ByteBuffer> {
        val outputs = emptyParameterMap()
        runSignatureLocked(parametersToMap(parameters), outputs, "restore")
        return parametersFromMap(outputs)
    }

    fun fit(
        epochs: Int = 1,
        batchSize: Int = 16,
        lossCallback: ((Int, List<Float>) -> Unit)? = null
    ): List<Double> {
        return trainSampleLock.write {
            if (trainingSamples.isEmpty()) {
                Log.w(TAG, "No training samples available.")
                lossCallback?.invoke(0, listOf(Float.NaN))
                return@write listOf(Double.NaN)
            }
            (1..epochs).map { epoch ->
                val losses = trainOneEpoch(batchSize)
                lossCallback?.invoke(epoch, losses)
                val averageLoss = losses.average()
                averageLoss
            }
        }
    }

    fun evaluate(): Pair<Float, Float> {
        val result = testSampleLock.read {
            if (testSamples.isEmpty()) {
                Log.w(TAG, "No test samples available for evaluation.")
                return@read Float.NaN to Float.NaN
            }
            val bottlenecks = testSamples.map { it.bottleneck }
            val logits = inference(spec.convertX(bottlenecks))
            spec.loss(testSamples, logits) to spec.accuracy(testSamples, logits)
        }
        Log.d(TAG, "Evaluate loss & accuracy: $result.")
        return result
    }

    fun inference(x: Array<X>): Array<Y> {
        val inputs = mapOf("x" to x)
        val logits = spec.emptyY(x.size)
        val outputs = mapOf("logits" to logits)
        runSignatureLocked(inputs, outputs, "infer")
        return logits
    }

    private fun trainOneEpoch(batchSize: Int): List<Float> {
        if (trainingSamples.isEmpty()) {
            Log.w(TAG, "No training samples available for this epoch.")
            return listOf(Float.NaN)
        }

        trainingSamples.shuffle()
        return trainingBatches(min(batchSize, trainingSamples.size)).map {
            val bottlenecks = it.map { sample -> sample.bottleneck }
            val labels = it.map { sample -> sample.label }
            training(spec.convertX(bottlenecks), spec.convertY(labels))
        }.toList()
    }

    private fun training(
        bottlenecks: Array<X>, labels: Array<Y>
    ): Float {
        val inputs = mapOf<String, Any>(
            "x" to bottlenecks,
            "y" to labels,
        )
        val loss = FloatBuffer.allocate(1)
        val outputs = mapOf<String, Any>(
            "loss" to loss,
        )
        runSignatureLocked(inputs, outputs, "train")
        return loss.get(0)
    }

    private fun trainingBatches(trainBatchSize: Int): Sequence<List<Sample<X, Y>>> {
        return sequence {
            var nextIndex = 0

            while (nextIndex < trainingSamples.size) {
                val fromIndex = nextIndex
                nextIndex += trainBatchSize

                val batch = if (nextIndex >= trainingSamples.size) {
                    trainingSamples.subList(
                        trainingSamples.size - trainBatchSize, trainingSamples.size
                    )
                } else {
                    trainingSamples.subList(fromIndex, nextIndex)
                }
                yield(batch)
            }
        }
    }

    fun parametersFromMap(map: Map<String, Any>): Array<ByteBuffer> {
        assertIntsEqual(layersSizes.size, map.size)
        return (0 until map.size).map {
            val buffer = map["a$it"] as ByteBuffer
            buffer.rewind()
            buffer
        }.toTypedArray()
    }

    fun parametersToMap(parameters: Array<ByteBuffer>): Map<String, Any> {
        assertIntsEqual(layersSizes.size, parameters.size)
        return parameters.mapIndexed { index, bytes -> "a$index" to bytes }.toMap()
    }

    private fun runSignatureLocked(
        inputs: Map<String, Any>,
        outputs: Map<String, Any>,
        signatureKey: String
    ) {
        interpreterLock.withLock {
            interpreter.runSignature(inputs, outputs, signatureKey)
        }
    }

    private fun emptyParameterMap(): Map<String, Any> {
        return layersSizes.mapIndexed { index, size -> "a$index" to ByteBuffer.allocate(size) }
            .toMap()
    }

    fun getTrainingSamplesCount(): Int = trainSampleLock.read { trainingSamples.size }
    fun getTestSamplesCount(): Int = testSampleLock.read { testSamples.size }

    companion object {
        private const val TAG = "Flower Client"
    }

    override fun close() {
        interpreter.close()
    }
}

data class Sample<X, Y>(val bottleneck: X, val label: Y)

class FakeNonEmptyMap<K, V> : HashMap<K, V>() {
    override fun isEmpty(): Boolean {
        return false
    }
}
