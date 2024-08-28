package flwr.android_client

//import dev.flower.flower_tflite.helpers.categoricalCrossEntropyLoss

import android.content.Context
import android.os.Handler
import android.text.TextUtils
import android.util.Log
import androidx.car.app.CarContext
import androidx.car.app.CarToast
import androidx.car.app.Screen
import androidx.car.app.model.Action
import androidx.car.app.model.CarColor
import androidx.car.app.model.CarIcon
import androidx.car.app.model.ItemList
import androidx.car.app.model.ListTemplate
import androidx.car.app.model.MessageTemplate
import androidx.car.app.model.Pane
import androidx.car.app.model.PaneTemplate
import androidx.car.app.model.Row
import androidx.car.app.model.Tab
import androidx.car.app.model.TabContents
import androidx.car.app.model.TabTemplate
import androidx.car.app.model.TabTemplate.TabCallback
import androidx.car.app.model.Template
import androidx.core.graphics.drawable.IconCompat
import dev.flower.flower_tflite.FlowerClient
import dev.flower.flower_tflite.SampleSpec
import dev.flower.flower_tflite.createFlowerService
import dev.flower.flower_tflite.helpers.classifierAccuracy
import dev.flower.flower_tflite.helpers.loadMappedAssetFile
import dev.flower.flower_tflite.helpers.negativeLogLikelihoodLoss
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.Runnable
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.LineNumberReader
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.random.Random

// TabScreen: shows the status of the engine, settings, and logs
class TabScreen(carContext: CarContext) : Screen(carContext) {
    object EngineStatusService {
        var isEngineOK: Boolean = true // Engine status, true if OK, false if not OK
            private set
    }

    private val mTabs: MutableMap<String, Tab> = HashMap() // Tabs map
    private val mTabContentsMap: MutableMap<String, TabContents> = HashMap() // Tab contents map
    private var mTabTemplateBuilder: TabTemplate.Builder? = null // Tab template builder
    private var mActiveContentId: String? = null // Active content ID

    private var flEnabled = false // Federated Learning enabled
    private var notificationsEnabled = false // Notifications enabled

    private val logs = StringBuilder() // Logs buffer

    private val handler = Handler()
    private var engineCheckRunnable: Runnable? = null

    private val scope = MainScope()
    private var flowerClient: FlowerClient<FeatureArray, FloatArray>? = null
    private var trainingJob: Job? = null // Job for managing the training process

    private var engineStatusText: String = "Engine is running smoothly"
    init {
        initializeEngineCheckRunnable()
    }

    override fun onGetTemplate(): Template {
        mTabTemplateBuilder = TabTemplate.Builder(object : TabCallback {
            override fun onTabSelected(tabContentId: String) {
                mActiveContentId = tabContentId
                invalidate()
            }
        }).setHeaderAction(Action.APP_ICON)

        mTabContentsMap.clear()
        mTabs.clear()

        for (i in ICON_RES_IDS.indices) {
            val contentId = i.toString()
            var contentTemplate = when (i) {
                0 -> createShortMessageTemplate()
                1 -> createPaneTemplate("Settings Content")
                2 -> createLogsTemplate()
                else -> createLogsTemplate()
            }
            val tabContents = TabContents.Builder(contentTemplate).build()
            mTabContentsMap[contentId] = tabContents

            val tabBuilder = Tab.Builder()
                .setTitle(TITLES[i])
                .setIcon(
                    CarIcon.Builder(
                        IconCompat.createWithResource(
                            carContext,
                            ICON_RES_IDS[i]
                        )
                    ).build()
                )
                .setContentId(contentId)
            if (TextUtils.isEmpty(mActiveContentId) && i == 0) {
                mActiveContentId = contentId
                mTabTemplateBuilder!!.setTabContents(tabContents)
            } else if (TextUtils.equals(mActiveContentId, contentId)) {
                mTabTemplateBuilder!!.setTabContents(tabContents)
            }

            val tab = tabBuilder.build()
            mTabs[tab.contentId] = tab
            mTabTemplateBuilder!!.addTab(tab)
        }
        return mTabTemplateBuilder!!
            .setActiveTabContentId(mActiveContentId!!)
            .build()
    }

    private fun createLogsTemplate(): ListTemplate {
        val listBuilder = ItemList.Builder()
        val logEntries = logs.toString().split("\n".toRegex()).dropLastWhile { it.isEmpty() }
            .toTypedArray()
        for (entry in logEntries) {
            if (entry.isNotEmpty()) {
                listBuilder.addItem(Row.Builder().setTitle(entry).build())
            }
        }
        return ListTemplate.Builder()
            .setSingleList(listBuilder.build())
            .build()
    }

    private fun buildRowForTemplate(title: CharSequence, clickable: Boolean): Row {
        val rowBuilder = Row.Builder()
            .setTitle(title)
        if (clickable) {
            rowBuilder.setOnClickListener {
                CarToast.makeText(
                    carContext, title,
                    CarToast.LENGTH_SHORT
                ).show()
            }
        }
        return rowBuilder.build()
    }

    private fun createShortMessageTemplate(): MessageTemplate {
        val isEngineOK = EngineStatusService.isEngineOK
        //val message =
        //    if (isEngineOK) "Engine is running smoothly" else "Problem: " + EngineStatusService.engineProblem

        val iconResId = if (isEngineOK) R.drawable.ic_engine else R.drawable.ic_engine
        val icon = CarIcon.Builder(IconCompat.createWithResource(carContext, iconResId)).build()

        return MessageTemplate.Builder(this.engineStatusText)
            .setIcon(icon)
            .build()
    }

    private fun createPaneTemplate(paneTitle: String): PaneTemplate {
        val paneBuilder = Pane.Builder()
            .setImage(
                CarIcon.Builder(
                    IconCompat.createWithResource(
                        carContext,
                        R.drawable.ic_logo
                    )
                ).build()
            )

        paneBuilder.addRow(
            Row.Builder()
                .setTitle("Informed Consent")
                .addText("This app uses Federated Learning to improve engine fault predictions, making your car safer and more reliable. By giving your consent, you allow your car to contribute to this effort without compromising your privacy. Only the insights learned from your car's data, not the raw data itself, are shared with others. You acknowledge that upon pressing the button, the federated learning process will start. Pressing the button again will stop the learning process only after the completion of the Ongoing training task.")
                .build()
        )

        // Federated Learning Button
        paneBuilder.addAction(
            SwitchFLAction("Federated Learning: ", flEnabled, object : ClickFLListener {
                override fun onSwitchClick(isOn: Boolean) {
                    if (flEnabled) {
                        flEnabled = !flEnabled
                    } else {
                        createFlowerClient()
                        scope.launch {
                            loadDataInBackground()
                        }
                        flEnabled = !flEnabled

                    }
                    logAction("[SYSTEM] Federated Learning switched to " + (if (isOn) "ON" else "OFF"))
                    invalidate()
                }
            })
        )

        // Notification Button
        paneBuilder.addAction(
            SwitchNotifyAction("Show Notification: ", notificationsEnabled, object : ClickNotifyListener {
                override fun onSwitchClick(isOn: Boolean) {
                    notificationsEnabled = !notificationsEnabled
                    logAction("[SYSTEM] Show Notification switched to " + (if (isOn) "ON" else "OFF"))
                    invalidate()
                }
            })
        )

        return PaneTemplate.Builder(paneBuilder.build())
            .build()
    }

    private fun initializeEngineCheckRunnable() {
        engineCheckRunnable = object : Runnable {
            override fun run() {
                val engineStatus = mockUpdateStatus()
                updateEngineStatusText(engineStatus)
                val logTag = if (engineStatus != 0) "" else "[ERROR]"
                logAction("$logTag $engineStatus")
                invalidate()
                handler.postDelayed(this, ENGINE_CHECK_INTERVAL.toLong())
            }
        }
        handler.post(engineCheckRunnable as Runnable) // Start periodic engine status check
    }

    private fun mockUpdateStatus() : Int {
        var engineStatus = 0
        val filePath = "data/test_1.csv"
        try {
            val randomLine = runBlocking {
                val line = readRandomCsvLine(carContext, filePath)
                Log.d("D", "Random line: $line")
                // Split the line into label (the first element) and features, all separated by commas and convert to FloatArray
                val parts = line?.split(",")?.map { it.toFloat() }?.toFloatArray()
                val legData = mutableListOf<FeatureArray>()
                //Exclude the first element, the label
                legData.add(parts?.copyOfRange(1, parts.size)!!)
                Log.d("D", "legData: $legData")
                val inferenceResult = flowerClient?.inference(flowerClient!!.spec.convertX(legData))
                val firstResult = inferenceResult?.get(0)
                val maxEntry = firstResult?.withIndex()?.maxWithOrNull { a, b -> a.value.compareTo(b.value) }
                val maxIndex = maxEntry?.index
                val maxValue = maxEntry?.value
                if (maxIndex != null) {
                    engineStatus = maxIndex
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return engineStatus
    }

    suspend fun readRandomCsvLine(context: Context, fileName: String): String? {
        return withContext(Dispatchers.IO) {
            val lineCount = countLines(context, fileName)
            Log.d("D", "Total line count: $lineCount")
            if (lineCount == 0) return@withContext null

            val randomLineIndex = Random.Default.nextInt(lineCount)
            Log.d("D", "Random line index: $randomLineIndex")
            var randomLine: String? = null
            context.assets.open(fileName).bufferedReader().use { reader ->
                repeat(randomLineIndex) { reader.readLine() } // Skip lines
                randomLine = reader.readLine()
            }
            Log.d("D", "Selected random line: $randomLine")
            randomLine
        }
    }

    // Helper function to efficiently count lines in a file (remains the same)
    private suspend fun countLines(context: Context, fileName: String): Int {
        return withContext(Dispatchers.IO) {
            BufferedReader(InputStreamReader(context.assets.open(fileName))).useLines { it.count() }
        }
    }


    private fun updateEngineStatusText(engineStatus: Int) {
        when (engineStatus) {
            0 -> {
                this.engineStatusText = "Engine Status: All checks OK. Normal Operating Conditions."
            }
            1 -> {
                this.engineStatusText = "Engine Status: Rich Mixture detected. Fault code: 1."
            }
            2 -> {
                this.engineStatusText = "Engine Status: Lean Mixture detected. Fault code: 2."
            }
            3 -> {
                this.engineStatusText = "Engine Status: Low voltage detected. Fault code: 3."
            }
            else -> {
                this.engineStatusText = "Engine Status: Unknown fault detected. Fault code: -1"
            }
        }
    }

    private fun logAction(action: String) {
        val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
        logs.append(action).append(" at [").append(timestamp).append("]\n")
    }

    private fun SwitchNotifyAction(
        title: String,
        isOn: Boolean,
        listener: ClickNotifyListener
    ): Action {
        return Action.Builder()
            .setTitle(title + " " + (if (isOn) "ON" else "OFF"))
            .setOnClickListener {
                val newIsOn = !isOn
                CarToast.makeText(
                    carContext,
                    title + " " + (if (newIsOn) "ON" else "OFF"),
                    CarToast.LENGTH_SHORT
                ).show()
                listener.onSwitchClick(newIsOn)
            }
            .build()
    }

    private fun SwitchFLAction(
        title: String,
        isOn: Boolean,
        listener: ClickFLListener
    ): Action {
        return Action.Builder()
            .setTitle(title + " " + (
                    if (isOn){
                        "ON";
                    } else {
                        "OFF";
                    }))
            .setOnClickListener {
                val newIsOn = !isOn
                CarToast.makeText(
                    carContext,
                    title + " " + (if (!newIsOn) "OFF" else "ON"),
                    CarToast.LENGTH_SHORT
                ).show()
                listener.onSwitchClick(newIsOn)
            }
            .build()
    }

    private fun createFabBackAction(): Action {
        return Action.Builder()
            .setIcon(CarIcon.BACK)
            .setBackgroundColor(CarColor.BLUE)
            .setOnClickListener { screenManager.pop() }
            .build()
    }

    private interface ClickNotifyListener {
        fun onSwitchClick(isOn: Boolean)
    }

    private interface ClickFLListener {
        fun onSwitchClick(isOn: Boolean)
    }

    companion object {
        private const val LIST_SIZE = 10

        private const val ENGINE_CHECK_INTERVAL =
            5000 // Engine check interval in milliseconds

        // Tab titles
        private val TITLES = arrayOf(
            "Engine",
            "Settings",
            "Logs"
        )

        // Icons for tabs
        private val ICON_RES_IDS = intArrayOf(
            R.drawable.ic_engine,
            R.drawable.ic_settings,
            R.drawable.ic_logs
        )
    }

    private fun createFlowerClient() {
        val buffer = loadMappedAssetFile(carContext, "model/enginefaultdb.tflite")
        val layersSizes = intArrayOf(504, 36, 144, 16)
        val sampleSpec = SampleSpec<FeatureArray, FloatArray>(
            { it.toTypedArray() },
            { it.toTypedArray() },
            { Array(it) { FloatArray(CLASSES.size) } },
            ::negativeLogLikelihoodLoss,
            ::classifierAccuracy,
        )
        flowerClient = FlowerClient(buffer, layersSizes, sampleSpec)
    }

    private suspend fun runWithStacktrace(call: suspend () -> Unit) {
        try {
            call()
        } catch (err: Error) {
            Log.e(TAG, Log.getStackTraceString(err))
        }
    }

    private suspend fun <T> runWithStacktraceOr(or: T, call: suspend () -> T): T {
        return try {
            call()
        } catch (err: Error) {
            Log.e(TAG, Log.getStackTraceString(err))
            or
        }
    }

    private suspend fun loadDataInBackground() {
        val result = runWithStacktraceOr("Failed to load training dataset.") {
            loadData(carContext, flowerClient!!, DEVICE_ID)
            "Training dataset is loaded in memory. Ready to train!"
        }
        CarToast.makeText(carContext, result, CarToast.LENGTH_SHORT).show()

        scope.launch {
            runGrpcInBackground(IP, PORT)
        }
    }

    private suspend fun runGrpcInBackground(host: String, port: Int) {
        val address = "dns:///$host:$port"
        runWithStacktraceOr("Failed to connect to the FL server \n") {
            createFlowerService(address, false, flowerClient!!) {
                //println("Result of Flower Service's creation: $it")
                CarToast.makeText(carContext, it, CarToast.LENGTH_SHORT).show()
            }
            "Connection to the FL server successful \n"
        }
    }

    private suspend fun runTraining() {
        runWithStacktrace {
            flowerClient?.fit()
            logAction("[TRAINING] Training completed.")
        }
    }
}

private const val TAG = "TabScreen"

typealias FeatureArray = FloatArray

private val DEVICE_ID = 0
private const val IP = "10.0.2.2"
private const val PORT = 8080
