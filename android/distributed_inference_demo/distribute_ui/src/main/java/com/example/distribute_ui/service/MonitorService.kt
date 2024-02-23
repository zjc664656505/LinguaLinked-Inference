package com.example.distribute_ui.service

//import com.example.distribute_ui.TAG
import android.app.ActivityManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.os.Binder
import android.os.IBinder
import android.util.Log
import com.example.SecureConnection.Config
import com.example.SecureConnection.Client
import com.example.distribute_ui.Log.Logger
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.json.JSONObject
import org.zeromq.SocketType
import org.zeromq.ZContext
import org.zeromq.ZMQ
import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.FileReader
import java.io.IOException
import java.io.InputStreamReader
import java.net.ConnectException
import java.net.Inet4Address
import java.net.NetworkInterface
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketException
import java.nio.ByteBuffer
import java.util.Properties


private const val TAG = "LinguaLinked_app"
private const val mTAG = "LinguaLinked_monitor"
interface MonitorActions {
    fun getLatencyAndBandwidth()
}

class MonitorService : Service(), MonitorActions{
    private lateinit var sharedPref: SharedPreferences
    private val serviceScope = CoroutineScope(Dispatchers.IO)
    private val monitorNetworkScope = CoroutineScope(Dispatchers.IO)
    private val serverScope = CoroutineScope(Dispatchers.IO)

    private var serverIPAddress : String? = null;
    private lateinit var ipList : List<String>

    private val binder = LocalBinder()
//    private var availMemory: Long? = null
    private var cpuFrequency: Double? = null

    private var monitorReady: Boolean = false
    private val latencyMap = HashMap<String, String>()
    private val bandwidthMap = HashMap<String, String>()
    private var role = "worker"
    private var ips = mutableListOf<String>()
    private var currentIP: String? = null
    private var ipMapIndex = HashMap<String, Int>()
    private var monitorSendCheck: Int? = null

    private var latencyArr: DoubleArray? = null
    private var bandwidthArr: LongArray? = null
    private var totalMemory: Long? = null
    private var availMemory: Long? = null
    private var flop: Double = 0.0
    private var flopNum: Long = 0

    private var monitorBreak: Boolean = false

    inner class LocalBinder : Binder() {
        fun getService(): MonitorService = this@MonitorService
    }


    override fun onBind(intent: Intent?): IBinder? {
        Log.d(TAG, "onBind function is called in monitor system")
        sharedPref = getSharedPreferences("myPrefs", Context.MODE_PRIVATE)
        with(sharedPref.edit()) {
            remove("ips")
            apply()
        }

//        val path = filesDir
//        Log.d(TAG, "path in service is $path")
        Logger.instance.log(this, "start logging")
        getDeviceAvailableRAM()
//        getAppRam()
//        getNetworkLatency()
//        getBandwidth()
//        getDeviceFrequency()
//        latencyScope.launch {
//            val latency = pingDevice(serverIPAddress)
//
//        }

        // Test P2P network bandwidth and latency
//        GlobalScope.launch(Dispatchers.IO) {
//            getP2PbandwidthAndLatency()
//        }


        Log.d(TAG, "memory, freq is collected")
        return binder
    }

    override fun onCreate() {
        super.onCreate()
        serverIPAddress = getServerIPAddressFromConfig()
        Log.d(TAG, "Server IP Address: $serverIPAddress")
    }

    private fun getServerIPAddressFromConfig(): String? {
        val properties = Properties()
        try {
            assets.open("config.properties").use { inputStream ->
                properties.load(inputStream)
                return properties.getProperty("server_ip")
            }
        } catch (e: IOException) {
            Log.e(TAG, "Failed to load server IP from config.properties", e)
        }
        return null
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "onStartCommand function is called in monitor system")
        var id = 0
        if (intent != null && intent.hasExtra("role")) {
            id = intent.getIntExtra("role", 0)
        }
        if (id == 1) {
            role = "header"
        }
        GlobalScope.launch(Dispatchers.IO) {
            Log.d(mTAG, "create startMonitor thread")
            startMonitorThread()
        }

        return START_STICKY
    }


    private fun startMonitorThread() {
        val monitorConfig = Config(serverIPAddress, 34567)
        currentIP = Config.local
        Log.d(mTAG, "current IP address is $currentIP")
        val monitorContext = ZContext()
//        val monitorPort = 56789
        val monitorSocket: ZMQ.Socket = monitorContext.createSocket(SocketType.DEALER)
        monitorSocket.connect("tcp://${monitorConfig.root}:${monitorConfig.rootPort}")

        val currentIP = Config.local
        val jsonObject = JSONObject()
        jsonObject.put("ip", currentIP)
        jsonObject.put("role", role)
        monitorSocket.sendMore("MonitorIP")
        monitorSocket.send(jsonObject.toString())
        Log.d(mTAG, "IP message sent in monitor")
        val ipGraphByte: ByteArray = monitorSocket.recv(0)
        val ipGraph = String(ipGraphByte, Charsets.UTF_8)
        Log.d(mTAG, "IP graph received, graph is $ipGraph")
        ips = ipGraph.split(",") as MutableList<String>
        monitorSendCheck = ips.size - 1
        Log.d(mTAG, ips.toString())
        for (i in ips.indices) {
            ipMapIndex[ips[i]] = i
        }
        receiveFlopInfo(monitorSocket)

        latencyArr = DoubleArray(ips.size)
        bandwidthArr = LongArray(ips.size)
        totalMemory = 0
        availMemory = 0

        serverScope.launch {
            Log.d(mTAG, "create monitor server thread")
            startServer()
        }

        while (true) {
            val signal = String(monitorSocket.recv(), Charsets.UTF_8)
            Log.d(mTAG, "receive the monitor signal from server, signal = $signal")
            if (signal == "stop") {
                monitorBreak = true
                break
            }
            monitorSendCheck = ips.size - 1
            getFlop()
            getDeviceLatency()
            Log.d(mTAG, "the latency results: ${latencyArr!!.joinToString (", ")}")
//            getAppRam()
            getDeviceAvailableRAM()
            getDeviceBandwidth()
            uploadMonitorInfo(monitorSocket)
        }

    }

    private fun uploadMonitorInfo(socket: ZMQ.Socket) {
        while (monitorSendCheck!! > 0) {
//            Log.d(mTAG, "monitorSendCheck = $monitorSendCheck")
//            Log.d(mTAG, "get stuck in upload monitor info")
        }
        monitorSendCheck = ips.size - 1

        val jsonObject = JSONObject()

        jsonObject.put("ip", currentIP)
        jsonObject.put("latency", latencyArr!!.toList())
        jsonObject.put("bandwidth", bandwidthArr!!.toList())
        Log.d(mTAG, "bandwidth is ${bandwidthArr.contentToString()}")
        jsonObject.put("memory", listOf(totalMemory, availMemory))
        jsonObject.put("flop", flop)
        socket.sendMore("Monitor")
        socket.send(jsonObject.toString())
        Log.d(mTAG, "Monitor message sent")
    }

    private fun getFlop() {
        val flopBinPath = "$filesDir/flop_byte_array.bin"
        Log.d(mTAG, "flopBinPath = $flopBinPath")
        val flopOnnxPath = "$filesDir/flop_test_module.onnx"
        val flop_test_tensor = loadByteArrayFromFile(flopBinPath)
        val ort_session_flop_test = createSession(flopOnnxPath)
        var device_flop_per_sec = modelFlopsPerSecond(flopNum.toInt(), ort_session_flop_test, flop_test_tensor)
        Log.d(mTAG, "Device Flops/second: $device_flop_per_sec")
        flop = device_flop_per_sec
        releaseSession(ort_session_flop_test)
    }

    private fun receiveFlopInfo(socket: ZMQ.Socket) {
        val byteArray = socket.recv()
        flopNum = byteArray.toLongLittleEndian()
        Log.d(mTAG, "flopNum = $flopNum")
        val flopBinPath = "$filesDir/flop_byte_array.bin"
        Log.d(mTAG, "flopBinPath = $flopBinPath")
        val flopOnnxPath = "$filesDir/flop_test_module.onnx"
        val clientInstance = Client()
        clientInstance.receiveModelFile(flopBinPath, socket, true, 1024 * 1024)
        clientInstance.receiveModelFile(flopOnnxPath, socket, true, 1024 * 1024)
    }

    private fun loadByteArrayFromFile(filePath: String): ByteArray {
        return File(filePath).readBytes()
    }


    private fun ByteArray.toLongLittleEndian(): Long {
        return foldIndexed(0L) { index, acc, byte -> acc or ((byte.toLong() and 0xFF) shl (8 * index)) }
    }

    private fun getDeviceLatency() {
        for (i in ips.indices) {
            if (ips[i] != currentIP) {
                Log.d(mTAG, "send ping message to ip ${ips[i]}")
                val latency = pingDevice(ips[i])
                Log.d(mTAG, "latency to ${ips[i]} is $latency")
                latencyArr!![i] = latency
            }
        }
    }

    private fun pingDevice(serverAddress: String?): Double {
//        Log.d(TAG, "Enter pingServer method")
        try {
            val command = "ping -c 3 $serverAddress"
            val process = Runtime.getRuntime().exec(command)
            val inputStream = process.inputStream
            val reader = BufferedReader(InputStreamReader(inputStream))
            val output = StringBuilder()
            var line: String?

            while (reader.readLine().also { line = it } != null) {
                output.append(line).append("\n")
            }
            reader.close()
            Log.d(TAG, "latency output is ${output.toString()}")

            val avg = extractAvgPing(output.toString().trimIndent()) / 1000
            Log.d(TAG, "the avg latency is $avg from device $serverAddress")
            return avg

        } catch (e: Exception) {
            Log.d(TAG, "error message: $e")
        }
        return -1.0 // Error or no time value found
    }

    private fun extractAvgPing(input: String): Double {
        // Regex pattern to match the "avg" value from the rtt line
        val regex = """rtt min/avg/max/mdev = \d+(?:\.\d+)?/\d+(?:\.\d+)?/\d+(?:\.\d+)?/\d+(?:\.\d+)? ms""".toRegex()

        // Find a match in the input string
        val matchResult = regex.find(input)

        // Split the rtt line by "=" and then by "/" to get individual rtt values
        val rttValues = matchResult?.value?.split("=")?.get(1)?.trim()?.split("/") ?: return 0.0

        // Return the "avg" value (2nd value in the rtt line) as a Double
        return rttValues.getOrNull(1)?.toDoubleOrNull()!!
    }

    private fun getDeviceAvailableRAM() {
        val activityManager : ActivityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        availMemory = memoryInfo.availMem / 1000000
        Log.d(mTAG, "available memory of the device is $availMemory")
        totalMemory = memoryInfo.totalMem / 1000000
        Log.d(mTAG, "total memory of the device is $totalMemory")
//        Logger.instance.log(this, "available memory of the device is ${memoryInfo.availMem}")
    }

    private fun getAppRam() {
        // get memory of APP JVM
        val runtime = Runtime.getRuntime()
        val totalMemory = runtime.totalMemory()
        val freeMemory = runtime.freeMemory()
        Log.d(mTAG, "total memory is $totalMemory")
        Log.d(mTAG, "free memory is $freeMemory")
    }

    private fun getDeviceBandwidth() {
        for (ip in ips) {
            if (ip != currentIP) {
                startClient(ip)
            }
        }
    }

    private fun getP2PbandwidthAndLatency() {
        GlobalScope.launch(Dispatchers.IO) {
            val ipSet = hashSetOf<String>("128.195.41.56", "128.195.41.55")
            val currentIP = getCurrentDeviceIP()
            Log.d(mTAG, "current IP is $currentIP")
            ipSet.remove(currentIP)


            val serverJob = serverScope.launch {
                startServer()
            }

            val monitorJob = monitorNetworkScope.launch {
                // latency test
                delay(2000L)
                for (ip in ipSet) {
                    val latency = pingDevice(ip)
                    val latencyStr = "Latency to ${ip}: $latency ms \n"
                    latencyMap[ip] = latencyStr
                    startClient(ip)
                }
            }
            serverJob.join()
            monitorJob.join()
            monitorReady = true
        }
        while(!monitorReady) {

        }
        for((key, value) in bandwidthMap) {
            Logger.instance.log(this, "bandwidth from $key is $value")
        }
        for((key, value) in latencyMap) {
            Logger.instance.log(this, value)
        }
    }

    private fun startClient(serverIP: String) {
        var attempt = 0
        val maxAttempts =10
        val delayMillis = 5000L
        var socket: Socket? = null
        while (true) {
            try {
                Log.d(mTAG, "trying to connect to server")
                socket = Socket(serverIP, 55555)
                Log.d(mTAG, "connect to " + serverIP + "successfully")
                break
            } catch (e: ConnectException) {
                Log.d(mTAG, "Attempt $attempt failed: ${e.message}. Retrying in ${delayMillis}ms...")
                Thread.sleep(delayMillis)
                attempt ++
            }
        }
//        if (attempt == maxAttempts) {
//            Log.d(mTAG, "Failed to connect after $maxAttempts attempts.")
//            return
//        }

//        val socket = Socket(serverIP, 9999)
//        Log.d(mTAG, "Connected to server!")

        val duration = 500 // 0.5 seconds

        val buffer = ByteArray(4096 * 4) { 0 } // dummy data

        val end = System.currentTimeMillis() + duration
        try {
            while (System.currentTimeMillis() < end) {
                socket!!.getOutputStream().write(buffer)
            }
//            socket!!.getOutputStream().write(buffer)
        } catch (e: Exception) {
            Log.d(mTAG, e.toString())
        }

        socket!!.close()
        Log.d(mTAG, "Data sent!")
    }

    private fun startServer() {
        val serverSocket = ServerSocket(55555)
        Log.d(mTAG, "Server listening on port 55555...")
        val serverBeginTime = System.currentTimeMillis()
//        val TIMEOUT = 100 * 1000
        while (true) {
            val socket = serverSocket.accept()
            // Spawn a new thread to handle the client connection
            Thread {
                handleClient(socket)
            }.start()

            if (monitorBreak) {
                break
            }

//            if (System.currentTimeMillis() - serverBeginTime > TIMEOUT) {
//                break
//            }
        }
    }

    private fun handleClient(socket: Socket) {
        Log.d(mTAG, "Client connected!")
        val ipSource = socket.inetAddress.hostAddress
        Log.d(mTAG, "source ip is $ipSource")

        val start = System.currentTimeMillis()
        val duration = 10000 // 10 seconds

        var bytesRead = 0L
        val buffer = ByteArray(4096)
        while (System.currentTimeMillis() - start < duration) {
            val read = socket.getInputStream().read(buffer)
            if (read == -1) break
            bytesRead += read
        }
        val actualDuration = System.currentTimeMillis() - start
        Log.d(mTAG, "bandwidth transmit time is: $actualDuration")
        Log.d(mTAG, "bytesread is: $bytesRead")
        val bandwidth = bytesRead / actualDuration // MB per second
        Log.d(mTAG, "bandwidth calcaulation: " + bandwidth)
        if (bandwidthArr != null) {
            Log.d(mTAG, "bandwidthArr: " + bandwidthArr)
        } else {
            Log.d(mTAG, "bandwidthArr is null")
            bandwidthArr = LongArray(ips.size)
        }

        if (ipMapIndex != null) {
            Log.d(mTAG, "ipMapIndex: " + ipMapIndex)
            if (!ipMapIndex.containsKey(ipSource)) {
                Log.d(mTAG, ipSource + " not in ipMapIndex")
            }
        } else {
            Log.d(mTAG, "ipMapIndex is null")
        }

        bandwidthArr!![ipMapIndex[ipSource]!!] = bandwidth
//        val bandStr = "Bandwidth: $bandwidth bytes/s"
        Log.d(mTAG, "Bandwidth from $ipSource: $bandwidth bytes/s")
        monitorSendCheck = monitorSendCheck!! - 1

//        bandwidthMap[ipSource] = bandStr

        socket.close()
    }

    private fun getCurrentDeviceIP(): String? {
        try {
            val en = NetworkInterface.getNetworkInterfaces()
            while (en.hasMoreElements()) {
                val networkInterface = en.nextElement()
                val enumIpAddr = networkInterface.inetAddresses
                while (enumIpAddr.hasMoreElements()) {
                    val inetAddress = enumIpAddr.nextElement()
                    if (!inetAddress.isLoopbackAddress && inetAddress is Inet4Address) {
                        return inetAddress.getHostAddress().toString()
                    }
                }
            }
        } catch (ex: SocketException) {
//            Log.e(ConfigCreator.TAG, String.format("Current Device IP %s", ex.toString()))
        }
        return null
    }

//    fun getAvailableMemory(): Long? {
//        return availMemory
//    }

    fun getFrequency(): Double? {
        return cpuFrequency
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "stop monitor service")
    }

    private fun getDeviceFrequency() {
        try {
            val reader = FileReader("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")

            val bufferedReader = BufferedReader(reader)
            val maxFreqString = bufferedReader.readLine()
            val freqInKHz = maxFreqString.trim { it <= ' ' }.toInt()
//            for (i in 1..5) {
//                Log.d(TAG, "test freq")
//                val reader = FileReader("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
//                val bufferedReader = BufferedReader(reader)
//                val maxFreqString = bufferedReader.readLine()
//                val freqInKHz = maxFreqString.trim { it <= ' ' }.toInt()
////            freqInKHz / 1000 // Convert kHz to MHz
//                Log.d(TAG, "freq is $freqInKHz")
//                bufferedReader.close()
//                reader.close()
//            }
            bufferedReader.close()
            reader.close()
//            freqInKHz / 1000 // Convert kHz to MHz
            Log.d(TAG, "max frequency is $freqInKHz")
            cpuFrequency = freqInKHz / 1000.0
        } catch (e: IOException) {
            e.printStackTrace()
            -1 // Handle the error accordingly
        }
    }

    private fun getBandwidth(serverIp: String) {
        Log.d(TAG, "getBandwidth")
        val SERVER_PORT = 8888
        val socket = Socket(serverIp, SERVER_PORT)
        val bytes = ByteArray(1024) { 0 }   // Dummy data
        for (i in 1..10000) {  // Send 10MB of data as an example
            socket.getOutputStream().write(bytes)
        }
        socket.close()
    }

    override fun getLatencyAndBandwidth() {
        Log.d(TAG, "start latency and bandwidth measurement")
        val timeout = 30
        val sharedPref = getSharedPreferences("myPrefs", Context.MODE_PRIVATE)
        val ips = sharedPref.getString("ips", null)
        Log.d(TAG, "ips for measuring latency and bandwidth is $ips]")
        if (ips != null) {
            ipList = ips.split(",")
        } else {
            Log.d(TAG, "no ip found in sharedpreference")
            return
        }
        serviceScope.launch {
            getNetworkLatency()
            getBandwidth("128.195.41.51")
        }
    }

    private fun getNetworkLatency() {
        var latencyList : MutableList<Double> = mutableListOf()
        val latency = pingDevice(serverIPAddress)
        Log.d(TAG, "latency to server is $latency")
        latencyList.add(latency)
        for (i in ipList.indices) {
            Log.d(TAG, "ip in $i is ${ipList[i]}")
            val l = pingDevice(ipList[i])
            Log.d(TAG, "latency to server is $latency")
            latencyList.add(l)
        }
    }

    external fun createSession(inference_model_path:String): Long

    external fun releaseSession(session: Long)

    external fun modelFlopsPerSecond(modelFlops: Int, session: Long, data: ByteArray?): Double

    companion object {
        init {
            System.loadLibrary("distributed_inference_demo")
        }
    }
}