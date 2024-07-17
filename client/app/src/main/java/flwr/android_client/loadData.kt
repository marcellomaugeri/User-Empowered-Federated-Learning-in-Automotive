package flwr.android_client

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import dev.flower.flower_tflite.FlowerClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.concurrent.ExecutionException

suspend fun readAssetLines(
    context: Context,
    fileName: String,
    call: suspend (Int, String) -> Unit
) {
    withContext(Dispatchers.IO) {
        BufferedReader(InputStreamReader(context.assets.open(fileName))).useLines {
            it.forEachIndexed { i, l -> launch { call(i, l) } }
        }
    }
}

/**
 * Load training data from disk.
 * If you want to use raw EngineFaultDB parameters as input,
 * you can run a MinMaxScaler with these parameters to normalize the dataset:
 * Minimum values for each feature: [4.530000e-01 3.820000e-01 2.580000e+00 4.650000e-01
 *  1.066452e+03 1.917000e+00 5.187000e+00 2.275700e+01 4.210000e-01
 *  1.787000e+00 8.649000e+00 2.030000e-01 6.950000e-01 1.021000e+01]
 * Maximum values for each feature: [4.547000e+00 4.048000e+00 1.537118e+03 3.394600e+01
 *  5.013402e+03 1.481000e+01 2.004300e+01 1.075390e+02 1.013200e+01
 *  9.756570e+02 1.512900e+01 1.151000e+00 1.149000e+00 1.689300e+01]
 */
@Throws
suspend fun loadData(
    context: Context,
    flowerClient: FlowerClient<FeatureArray, FloatArray>,
    device_id: Int
) {
    if(device_id == 0) {
        readAssetLines(context, "data/train_1.csv") { index, line ->
            addSample(context, flowerClient, line, true)
        }
        readAssetLines(context, "data/test_1.csv") { index, line ->
            addSample(context, flowerClient, line, false)
        }
    } else if(device_id == 1) {
        readAssetLines(context, "data/train_2.csv") { index, line ->
            addSample(context, flowerClient, line, true)
        }
        readAssetLines(context, "data/test_1.csv") { index, line ->
            addSample(context, flowerClient, line, false)
        }
    } else if (device_id == 2) {
        readAssetLines(context, "data/test_1.csv") { index, line ->
            addSample(context, flowerClient, line, false)
        }
    }
}

@Throws
private fun addSample(
    context: Context,
    flowerClient: FlowerClient<FeatureArray, FloatArray>,
    row: String,
    isTraining: Boolean
) {
    val parts = row.split(",".toRegex())
    val index = parts[0].toInt()
    val className = CLASSES[index]
    val labelArray = classToArray(className)
    val features = parts.subList(1, parts.size).map { it.toFloat() }.toFloatArray()

    // add to the list.
    try {
        flowerClient.addSample(features, labelArray, isTraining)
    } catch (e: ExecutionException) {
        throw RuntimeException("Failed to add sample to model", e.cause)
    } catch (e: InterruptedException) {
        // no-op
    }
}

fun classToArray(className: String): FloatArray {
    return CLASSES.map {
        if (className == it) 1f else 0f
    }.toFloatArray()
}

const val INPUT_LAYER_SIZE = 14

val CLASSES = listOf(
    "NO FAULT",
    "RICH MIXTURE",
    "LEAN MIXTURE",
    "LOW VOLTAGE",
)

//CSV Fault,MAP,TPS,Force,Power,RPM,Consumption L/H,Consumption L/100KM,Speed,CO,HC,CO2,O2,Lambda,AFR
val FEATURES = listOf(
    "Fault",
    "MAP",
    "TPS",
    "Force",
    "Power",
    "RPM",
    "Consumption L/H",
    "Consumption L/100KM",
    "Speed",
    "CO",
    "HC",
    "CO2",
    "O2",
    "Lambda",
    "AFR"
)
