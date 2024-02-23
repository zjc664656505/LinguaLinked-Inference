package com.example.distribute_ui.Log

import android.content.Context
import android.os.Build
import android.os.Environment
import android.util.Log
import androidx.annotation.RequiresApi
import kotlinx.coroutines.CoroutineScope
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

private const val mTAG = "monitor"

class Logger private constructor() {

    fun log(context: Context, message: String) {
        val time = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            getCurrentTime()
        } else {
            "no_valid_time"
        }
//        val dirPath = Environment.getExternalStorageDirectory().absolutePath + "/LinguaLinked"
        val dirPath = context.getExternalFilesDir(null)
        if (dirPath == null) {
            Log.d(mTAG, "app log dir path is null")
            return
        }
        Log.d(mTAG, "dirPath is $dirPath")
        val monitorDir = dirPath.absolutePath + "/Monitor"
        val logFilePath = "$monitorDir/${time}_monitor.txt"
        try {
            val directory = File(monitorDir)
            if (!directory.exists()) {
                directory.mkdirs()
            }
            Log.d(mTAG, "dir is ready")
            val logFile = File(logFilePath)
            if (!logFile.exists()) {
                logFile.createNewFile()
            }
            Log.d(mTAG, "filePath is ready")
            val writer = FileWriter(logFile, true)
            writer.append(message + "\n")
            writer.flush()
            writer.close()
            Log.d(mTAG, "writing is ready")
        } catch (e: IOException) {
            Log.d(mTAG, e.toString())
        }
    }

    @RequiresApi(Build.VERSION_CODES.O)
    private fun getCurrentTime(): String {
        val current = LocalDateTime.now()

        val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd-H-m")
        return current.format(formatter)
    }

    companion object {
        // The single instance of Logger
        val instance: Logger by lazy { Logger() }
    }
}