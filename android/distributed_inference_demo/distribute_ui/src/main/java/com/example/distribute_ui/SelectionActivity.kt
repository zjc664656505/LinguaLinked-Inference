package com.example.distribute_ui

import android.Manifest
import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.example.distribute_ui.service.MonitorService
import com.example.distribute_ui.ui.InferenceViewModel
import com.example.distribute_ui.ui.theme.Distributed_inference_demoTheme


const val TAG = "LinguaLinked_app"

// Define the permission request code
private const val MY_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE = 1 // or any other unique integer


class SelectionActivity : ComponentActivity(), LatencyMeasurementCallbacks {

    private var monitorIntent: Intent? = null
    private var backgroundIntent: Intent? = null

    private val viewModel : InferenceViewModel by viewModels()

    private var service: MonitorService? = null
    private var serviceBound = false

    private var id = 0 // 1 -> header, 0 -> worker
    private var modelName = ""

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(className: ComponentName, iBinder: IBinder) {
            Log.d(TAG, "monitor service connection is successful")

//            val binder = service as MonitorActions.MyBinder
//            service = binder.getService()

            val binder = iBinder as MonitorService.LocalBinder
            service = binder.getService()
            serviceBound = true

            // Fetch data from service and update the ViewModel, upload memory and CPU frequency
//            val memory = service?.getAvailableMemory()
//            val freq = service?.getFrequency()
//            viewModel.prepareUploadData(memory ?: 0, freq ?: 0.0)

        }

        override fun onServiceDisconnected(arg0: ComponentName) {
            serviceBound = false
        }
    }

    private val receiver: BroadcastReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            Log.d(TAG, "selectionActivity receives the broadcast")
            monitorIntent!!.putExtra("role", id)
            startService(monitorIntent)
        }

    }

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Log.d(BackgroundService.TAG, "write external storage denied")
            // Permission is not granted
            // Should we show an explanation?
            if (ActivityCompat.shouldShowRequestPermissionRationale(
                    this,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
                )
            ) {
                // Show an explanation to the user *asynchronously* -- don't block
                // this thread waiting for the user's response! After the user
                // sees the explanation, try again to request the permission.
            } else {
                // No explanation needed; request the permission
                ActivityCompat.requestPermissions(
                    this, arrayOf<String>(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                    MY_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE
                )

                // MY_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE is an
                // app-defined int constant. The callback method gets the
                // result of the request.
            }
        } else {
            // Permission has already been granted
            Log.d(BackgroundService.TAG, "write external storage permit is ok")
        }

        backgroundIntent = Intent(this, BackgroundService::class.java)

        val filter = IntentFilter("START_MONITOR")
        LocalBroadcastManager.getInstance(this).registerReceiver(receiver, filter)
//        registerReceiver(receiver, filter)
        monitorIntent = Intent(this ,MonitorService::class.java)
//        startService(monitorIntent)

//        val activityManager : ActivityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
//        val memoryInfo = ActivityManager.MemoryInfo()
//        activityManager.getMemoryInfo(memoryInfo)
//        Log.d(TAG, "available memory is ${memoryInfo.availMem}")

        setContent {
            Distributed_inference_demoTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    HomeScreen(
                        onMonitorStarted = {
                            monitorIntent!!.putExtra("role", id)
                            startService(monitorIntent)
                        },
                        onBackendStarted = {
                            if (!BackgroundService.isServiceRunning) {
                                backgroundIntent!!.putExtra("role", id)
                                backgroundIntent!!.putExtra("model", modelName)
                                startService(backgroundIntent)
                            }
                        },
                        onModelSelected = {
                            setModel(it)
                        },
                        viewModel = viewModel,
                        onRolePassed = {
                            setRole(it)
                        }
                    )
                }
            }
        }

//        bindService(Intent(this, MonitorService::class.java), serviceConnection, Context.BIND_AUTO_CREATE)

    }

    private fun setRole(id: Int) {
        this.id = id
        Log.d(TAG, "id is $id")
    }

    private fun setModel(modelName: String) {
        this.modelName = modelName
        Log.d(TAG, "model name is $modelName")
    }

    override fun onDestroy() {
        super.onDestroy()
//        unregisterReceiver(receiver);
        LocalBroadcastManager.getInstance(this).unregisterReceiver(receiver)
        if (serviceBound) {
            unbindService(serviceConnection)
            serviceBound = false
        }
//        stopService(monitorIntent)
        stopService(backgroundIntent)
    }

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onLatencyMeasured(latency: Double) {
        viewModel.updateLatency(latency)
    }

    companion object {
        init {
            System.loadLibrary("distributed_inference_demo")
        }
    }

    external fun createSession(inference_model_path:String): Long
    external fun modelFlopsPerSecond(modelFlops: Int, session: Long, data: ByteArray?): Double
}

interface LatencyMeasurementCallbacks {
    fun onLatencyMeasured(latency: Double)
}

//@Preview(showBackground = true)
//@Composable
//fun GreetingPreview() {
//    Distributed_inference_demoTheme {
//        HomeScreen(viewModel = InferenceViewModel(application = null))
//    }
//}

