package com.example.distribute_ui.service

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import com.example.SecureConnection.Communication
import com.example.SecureConnection.Config
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException

class InferenceService() : Service() {
    private val TAG = "LinguaLinked_app "
    private val testInput = "I love deep learning"

    var config: Config? = null
    var com: Communication? = null

    override fun onBind(intent: Intent?): IBinder? {
        TODO("Not yet implemented")
    }

    override fun onCreate() {
        super.onCreate()
    }

    override fun onDestroy() {
        super.onDestroy()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "start service")
        val updateInfo = intent?.getStringArrayExtra("updateInfo")
        val filePath = intent?.getStringExtra("filePath")

        val serviceScope = CoroutineScope(Dispatchers.IO)
        serviceScope.launch {
            withContext(Dispatchers.IO) {
//                val configCreator = ConfigCreator()
//                configCreator.setRootServerIP("128.195.41.39")
//                Log.d(TAG, "set root server IP")
////                configCreator.sendIPToServer(updateInfo!![0], updateInfo[1])
//                Log.d(TAG, "send IP to server")
//                config = configCreator.createConfig()
//                var logNum = 1
//                while (config == null) {
//                    if (logNum > 0) {
//                        Log.d(TAG, "config is null, waiting for IPs from server")
//                        logNum -= 1
//                    }
//
//                }
//                com = Communication(config)
//                com!!.param.modelPath = "$filePath/module.onnx"
//                com!!.param.tokenizerPath = "$filePath/tokenizer.json"
//                try {
//                    com!!.prepare()
//                } catch (e: IOException) {
//                    Log.d(TAG, "IOException is $e")
//                    throw RuntimeException(e)
//                } catch (e: Exception) {
//                    Log.d(TAG, "exception is $e")
//                }
//                com!!.param.classes = arrayOf("Negative", "Positive")
//
//                val testInputs = arrayOf(
////                    input
//                    testInput
//                )
//                Log.d(TAG, "input is ${testInputs[0]}")
//                val corePoolSize = 2
//                val maximumPoolSize = 2
//                val keepAliveTime = 1000
//
//                try {
//                    com!!.running(corePoolSize, maximumPoolSize, keepAliveTime, testInputs)
//                } catch (e: IOException) {
//                    Log.d(TAG, "Inference IO exception is $e")
//                    throw java.lang.RuntimeException(e)
//                } catch (e: InterruptedException) {
//                    Log.d(TAG, "InterruptedException exception is $e")
//                    throw java.lang.RuntimeException(e)
//                } catch (e: Exception) {
//                    Log.d(TAG, "The exception is $e")
//                }
            }
        }


//        val executor = Executors.newSingleThreadExecutor()
//        executor.submit<Any?> {
//            val configCreator = ConfigCreator()
//            configCreator.setRootServerIP("128.195.41.39")
//            Log.d(TAG, "set root server IP")
//            configCreator.sendIPToServer(updateInfo!![0], updateInfo[1])
//            Log.d(TAG, "send IP to server")
//            config = configCreator.createConfig()
//            var logNum = 1
//            while (config == null) {
//                if (logNum > 0) {
//                    Log.d(TAG, "config is null, waiting for IPs from server")
//                    logNum -= 1
//                }
//
//            }
//            com = Communication(config)
//            com!!.param.modelPath = "$filePath/module.onnx"
//            com!!.param.tokenizerPath = "$filePath/tokenizer.json"
//            try {
//                com!!.prepare()
//            } catch (e: IOException) {
//                Log.d(TAG, "IOException is $e")
//                throw RuntimeException(e)
//            } catch (e: Exception) {
//                Log.d(TAG, "exception is $e")
//            }
//            com!!.param.classes = arrayOf("Negative", "Positive")
//
//        }
//
//        executor.submit<Any?> {
//            val testInputs = arrayOf(
////                    input
//                testInput
//            )
//            Log.d(TAG, "input is ${testInputs[0]}")
//            val corePoolSize = 2
//            val maximumPoolSize = 2
//            val keepAliveTime = 1000
//
//            try {
//                com!!.running(corePoolSize, maximumPoolSize, keepAliveTime, testInputs)
//            } catch (e: IOException) {
//                Log.d(TAG, "Inference IO exception is $e")
//                throw java.lang.RuntimeException(e)
//            } catch (e: InterruptedException) {
//                Log.d(TAG, "InterruptedException exception is $e")
//                throw java.lang.RuntimeException(e)
//            } catch (e: Exception) {
//                Log.d(TAG, "The exception is $e")
//            }
//        }

        return START_STICKY
    }
}