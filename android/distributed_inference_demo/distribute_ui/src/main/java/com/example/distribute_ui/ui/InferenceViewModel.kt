package com.example.distribute_ui.ui
import android.app.Application
import android.content.Context
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.example.SecureConnection.Communication
import com.example.SecureConnection.Config
import com.example.distribute_ui.DataRepository
import com.example.distribute_ui.TAG
import com.example.distribute_ui.data.modelMap
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter


@RequiresApi(Build.VERSION_CODES.O)
class InferenceViewModel(application: Application) : AndroidViewModel(application) {
    private var filesDirPath : String = ""
    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()
    val isDirEmpty: LiveData<Boolean> = DataRepository.isDirEmptyLiveData
    val sampleId: LiveData<Int> = DataRepository.sampleId
    private val _prepareState = MutableStateFlow(false)
    val prepareState: StateFlow<Boolean> = _prepareState.asStateFlow()
    val sharedPref = application.getSharedPreferences("myPrefs", Context.MODE_PRIVATE)
    var testInput = "I love deep learning"
    val chatHistory = _uiState.value.chatHistory
    private var updatingDecodedString: Boolean = false
    private var decodedStringIndex: Int = -1
    private var previousSampleID: Int = 0
    val modelAuhthor = "LinguaLinked"

    init {
        // Observe decodedStringLiveData from data repository
        DataRepository.decodingStringLiveData.observeForever { decodedString ->
            // Update chat history when decodedString changes
            if (!decodedString.isNullOrEmpty()) {
                if (updatingDecodedString) {
                    chatHistory.add(Messaging(modelAuhthor, decodedString, getCurrentFormattedTime()))
                    decodedStringIndex = chatHistory.lastIndex
                    updatingDecodedString = false
                } else {
                    // Otherwise, replace the previous decoded string with the updated one
                    if (decodedStringIndex != -1) {
                        chatHistory[decodedStringIndex!!] = Messaging(modelAuhthor, decodedString, getCurrentFormattedTime())
                    } else {
                        chatHistory.add(Messaging(modelAuhthor, decodedString, getCurrentFormattedTime()))
                        decodedStringIndex = chatHistory.lastIndex
                    }
                }
            }
        }
        DataRepository.sampleId.observeForever { sampleId ->
            // If sampleId is updated, set updatingDecodedString to true
            if (sampleId != null && sampleId != previousSampleID && chatHistory.size==decodedStringIndex+2) {
                // Set updatingDecodedString to true only when sampleId meets the conditions
                updatingDecodedString = true
                previousSampleID = sampleId
            }
        }
    }

    var nodeId: Int = 0
    var modelName: String = ""
    var config: Config? = null
    var com: Communication? = null
    private val job =  SupervisorJob()
    private val ioScope by lazy { CoroutineScope(job + Dispatchers.IO) }
    private val comScope by lazy { CoroutineScope(job + Dispatchers.IO) }
    private val downloadScope by lazy { CoroutineScope(job + Dispatchers.IO) }

    // monitor system service
    private val _availableMemory = MutableLiveData<Long>()
    val memory: LiveData<Long> get() = _availableMemory

    private val _cpuFrequency = MutableLiveData<Double>()
    private val _latency = MutableLiveData<Double>()

    fun prepareUploadData(memory: Long, freq: Double) {
        Log.d(TAG, "update memory in viewModel")
        _availableMemory.postValue(memory)
        _cpuFrequency.postValue(freq)
    }

    @RequiresApi(Build.VERSION_CODES.O)
    fun getCurrentFormattedTime(): String {
        val currentTimeMillis = System.currentTimeMillis()
        val instant = Instant.ofEpochMilli(currentTimeMillis)
        val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss").withZone(ZoneId.systemDefault())
        return formatter.format(instant)
    }

    fun updateLatency(latency: Double) {
        Log.d(TAG, "update latency is $latency")
        _latency.postValue(latency)
    }

    fun setDirPath(path : String) {
        this.filesDirPath = path
    }

    fun selectModel(modelName: String) {
        this.modelName = modelName
        _uiState.value.modelName = modelName
    }

    fun addChatHistory(msg: Messaging){
        chatHistory.add(msg)
    }

    private fun getUpdateInfo(): List<String> {
        val nodeString = if (nodeId == 0) "worker" else "header"
        val modelString: String = if (nodeId == 0) "" else modelMap.getOrDefault(this.modelName, "")
        return listOf<String>(nodeString, modelString)
    }

    fun resetOption() {
        this.nodeId = 0
        this.modelName = ""
    }

    private fun saveLatencyDevices(latencyDevices: String) {
        with(sharedPref.edit()) {
            putString("ips", latencyDevices)
            apply()
        }
    }

    fun updateDeviceInfo() {
        ioScope.launch {
            withContext(Dispatchers.IO) {
//                saveLatencyDevices("128.0.0.1,128.0.0.2")   // test sharedpreference

//                Log.d(TAG, "available memory is ready, the content is ${_availableMemory.value}")
//                val configCreator = ConfigCreator()
//                Log.d(TAG, "get config")
//                configCreator.setRootServerIP("128.195.41.39")
//                Log.d(TAG, "set root server")
//                val updateInfo = getUpdateInfo()
//                configCreator.sendIPToServer(updateInfo[0], updateInfo[1], _availableMemory.value, _cpuFrequency.value, _latency.value)
//                Log.d(TAG, "send IP to server")
//                config = configCreator.createConfig()
//                val latencyDevices = (config!!.prevNodes + config!!.nextNodes).toSet().joinToString(separator = ",")
//                Log.d(TAG, "latency devices : $latencyDevices")
//                saveLatencyDevices(latencyDevices)
            }
        }
    }


    // This function is depreciated
    fun inferenceExecution(content : String) {
        Log.d(TAG, "Enter inferenceExecution")
        this.testInput = content
        comScope.launch {
            withContext(Dispatchers.IO) {
                val testInputs = ArrayList(arrayOf(
                    testInput
                ).toList())
//                Log.d(TAG, "input is ${testInputs[0]}")
                val corePoolSize = 2
                val maximumPoolSize = 2
                val keepAliveTime = 1000

//                var logNum = 1
//                while (prepared == 0) {
//                    if (logNum > 0) {
//                        Log.d(TAG, "not prepared, the running is waiting")
//                        logNum -= 1
//                    }
//                }

                try {
                    com!!.running(corePoolSize, maximumPoolSize, keepAliveTime, testInputs)
                } catch (e: IOException) {
                    Log.d(TAG, "Inference IO exception is $e")
                    throw java.lang.RuntimeException(e)
                } catch (e: InterruptedException) {
                    Log.d(TAG, "InterruptedException exception is $e")
                    throw java.lang.RuntimeException(e)
                } catch (e: Exception) {
                    Log.d(TAG, "The exception is $e")
                }
//                val result = com!!.param.classesToFetch[0];
//                addMessage(
//                    Message("robot", result, "now")
//                )

            }
        }
    }

    fun inferencePrepare() {
        downloadScope.launch {
            withContext(Dispatchers.IO) {
                var logNum = 1
                while (config == null) {
                    if (logNum > 0) {
                        Log.d(TAG, "config is null, waiting for IPs from server")
                        logNum -= 1
                    }

                }
                com = Communication(config)
                com!!.param.modelPath = "$filesDirPath/module.onnx"
                try {
                    Log.d(TAG, "start prepare")
                    com!!.prepare()
                } catch (e: IOException) {
                    throw RuntimeException(e)
                }
                com!!.param.classes = arrayOf("Negative", "Positive")
//                prepared = 1
                if (nodeId == 0) {
                    val testInputs = ArrayList(arrayOf(
                        testInput
                    ).toList())
//                    Log.d(TAG, "input is ${testInputs[0]}")
                    val corePoolSize = 2
                    val maximumPoolSize = 2
                    val keepAliveTime = 1000
                    try {
                        Log.d(TAG, "worker start running")
                        com!!.running(corePoolSize, maximumPoolSize, keepAliveTime, testInputs)
                    } catch (e: IOException) {
                        Log.d(TAG, "Inference IO exception is $e")
                        throw java.lang.RuntimeException(e)
                    } catch (e: InterruptedException) {
                        Log.d(TAG, "InterruptedException exception is $e")
                        throw java.lang.RuntimeException(e)
                    } catch (e: Exception) {
                        Log.d(TAG, "The exception is $e")
                    }
                }

                if (nodeId == 1) {
                    _prepareState.value = true
                    Log.d(TAG, "preparestate is ${_prepareState.value}")
                }
//                _IPState.value = true

//                val testInput = arrayOf(
//                    "I love machine learning",
//                    "I hate machine learning",
//                    "I love distributed learning on edge!",
//                    "I hate distributed learning on edge!"
//                )
            }
        }



    }

    fun testPrepareState() {
        _prepareState.value = true
        Log.d(TAG, "preparestate is ${_prepareState.value}")
        val time = System.currentTimeMillis()
        Log.d(TAG, "time in testPrepareState is $time")
    }

    fun resetPrepareState() {
        _prepareState.value = false
    }

    init {
        Log.d(TAG, "InferenceViewModel init")
    }
}