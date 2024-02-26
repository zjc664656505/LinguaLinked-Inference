package com.example.distribute_ui

import android.net.wifi.WifiManager
import android.os.Build
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.dimensionResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.distribute_ui.data.exampleModelName
import com.example.distribute_ui.ui.ChatScreen
import com.example.distribute_ui.ui.InferenceViewModel
import com.example.distribute_ui.ui.ModelSelectionScreen
import com.example.distribute_ui.ui.NodeSelectionScreen
import com.example.distribute_ui.ui.theme.Distributed_inference_demoTheme
import kotlinx.coroutines.*

enum class AppScreen() {
    NodeSelection,
    ModelSelection,
    Chat
}

@RequiresApi(Build.VERSION_CODES.O)
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    navController: NavHostController = rememberNavController(),
    onMonitorStarted: () -> Unit,
    onBackendStarted: () -> Unit,
    onModelSelected: (modelName: String) -> Unit,
    viewModel: InferenceViewModel,
    onRolePassed: (id: Int) -> Unit
) {
    val isLowLatencyOn = remember { mutableStateOf(false) }
    val context = LocalContext.current
    val wifiManager = context.applicationContext.getSystemService(ComponentActivity.WIFI_SERVICE) as WifiManager
    val wifiLock = wifiManager.createWifiLock(
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) WifiManager.WIFI_MODE_FULL_LOW_LATENCY
        else WifiManager.WIFI_MODE_FULL_HIGH_PERF,
        "mylock"
    )

    Scaffold(
        topBar = {
            AppBar(
                onClick = {
                    isLowLatencyOn.value = !isLowLatencyOn.value
                    if (isLowLatencyOn.value) {
                        if (wifiLock?.isHeld == false) {
                            wifiLock?.acquire()
                            Toast.makeText(context, "Low latency mode is on", Toast.LENGTH_SHORT).show()
                            Log.d(TAG, "acquire wifiLock")
                        }
                    } else {
                        if (wifiLock?.isHeld == true) {
                            wifiLock?.release()
                            Toast.makeText(context, "Low latency mode is off", Toast.LENGTH_SHORT).show()
                            Log.d(TAG, "release wifiLock")
                        }
                    }
                },
                isLowLatencyOn.value
            )
        }
    ) { contentPadding ->
//        val uistate by viewModel.uiState.collectAsState()

        NavHost(
            navController = navController,
            startDestination = AppScreen.NodeSelection.name,
            modifier = Modifier.padding(contentPadding)
        ) {
            Log.d(TAG, "recompose in NavHost")
            composable(route = AppScreen.NodeSelection.name) {
                NodeSelectionScreen(
                    viewModel = viewModel,
                    onHeaderClicked = {
//                        viewModel.nodeId = 1
                    },
                    onWorkerClicked = {
//                        viewModel.nodeId = 0
                    },
                    onCancelClicked = {
                        viewModel.resetOption()
                    },
                    onNextClicked = {

                        onRolePassed(it)
                        if (it == 1) {
                            navController.navigate(AppScreen.ModelSelection.name)
                        }
                    },
                    onMonitorStarted = onMonitorStarted,
                    onBackendStarted = onBackendStarted,
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(dimensionResource(R.dimen.padding_medium))
                )
            }
            composable(route = AppScreen.ModelSelection.name) {
                ModelSelectionScreen(
                    viewModel = viewModel,
                    exampleModelName,
                    onCancelClicked = {
                        navController.navigate(AppScreen.NodeSelection.name)
                        viewModel.resetOption()
                    },
                    onNextClicked = {
                        Log.d(TAG, "Navigating to Chat screen")
                        navController.navigate(AppScreen.Chat.name)
                    },
                    onBackendStarted = {
                        onBackendStarted()
                    },
                    onModelSelected = {
                        onModelSelected(it)
                    },
                    modifier = Modifier.fillMaxHeight()
                )
            }
            composable(route = AppScreen.Chat.name) {
                ChatScreen(
                    viewModel = viewModel
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppBar(
    onClick: () -> Unit,
    isLowLatencyOn: Boolean,
    modifier: Modifier = Modifier
) {
    TopAppBar(
        title = { Text(stringResource(id = R.string.app_name)) },
        colors = TopAppBarDefaults.mediumTopAppBarColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        ),
        modifier = modifier,
        actions = {
            IconButton(onClick = {
                onClick()
            }) {
                Icon(
                    painter = if (!isLowLatencyOn) painterResource(id = R.mipmap.baseline_toggle_off_white_48) else painterResource(
                        id = R.mipmap.baseline_toggle_on_white_48
                    ),
                    contentDescription = "latency mode"
                )
            }
        }
    )
}

@Preview(showBackground = true)
@Composable
fun GameScreenPreview() {
    Distributed_inference_demoTheme {
//        HomeScreen()
    }
}
