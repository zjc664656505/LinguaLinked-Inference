package com.example.distribute_ui.ui

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.SharedPreferences
import android.util.Log
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.selection.selectable
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.dimensionResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.example.distribute_ui.BackgroundService
import com.example.distribute_ui.Events
import com.example.distribute_ui.R
import com.example.distribute_ui.TAG
import com.example.distribute_ui.ui.components.ButtonBar
import com.example.distribute_ui.ui.theme.Distributed_inference_demoTheme
import org.greenrobot.eventbus.EventBus
import org.greenrobot.eventbus.Subscribe
import org.greenrobot.eventbus.ThreadMode
import androidx.compose.runtime.livedata.observeAsState

@Composable
fun ModelSelectionScreen(
    viewModel: InferenceViewModel,
    options: List<String>?,
    onCancelClicked: () -> Unit,
    onNextClicked: () -> Unit,
    onBackendStarted: () -> Unit,
    onModelSelected: (modelName: String) -> Unit,
    modifier: Modifier = Modifier
){
    var selectedModel by remember { mutableStateOf("") }
    var selectedValue = remember { mutableStateOf(false) }
    val nextClickedState = remember { mutableStateOf(false) }
    val prepareState by viewModel.prepareState.collectAsState()
    val context = LocalContext.current
    val isDirEmpty by viewModel.isDirEmpty.observeAsState(initial = true)

    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        Column(modifier = Modifier.padding(dimensionResource(R.dimen.padding_medium))){
            options?.forEach { item ->
                Row(
                    modifier = Modifier.selectable(
                        selected = selectedModel == item,
                        onClick = {
                            selectedModel = item
                            selectedValue.value = true
                            onModelSelected(item)
                        }
                    ),
                    verticalAlignment = Alignment.CenterVertically
                ){
                    RadioButton(
                        selected = selectedModel == item,
                        onClick = {
                            selectedModel = item
                            Log.d(TAG, "selected Model is $selectedModel")
                            selectedValue.value = true
                            onModelSelected(item)
                        }
                    )
                    Text(item)
                }
            }
            Divider(
                thickness = dimensionResource(R.dimen.thickness_divider),
                modifier = Modifier.padding(bottom = dimensionResource(R.dimen.padding_medium))
            )
            Text(
                text = "Please select your model for execution.",
                textAlign = TextAlign.Left,
                style = MaterialTheme.typography.bodyMedium
            )
        }
        ButtonBar(
            modifier = modifier,
            onNextClicked = {
                nextClickedState.value = true
            },
            onCancelClicked = {
                nextClickedState.value = false
                selectedValue.value = false
                onCancelClicked()
            },
            selectedValue = selectedValue.value
        )
        if (nextClickedState.value && selectedValue.value) {
            Log.d(TAG, "Model Directory Path is Empty: " + isDirEmpty)
            onBackendStarted()
            HeaderProgressDialog(
                isEnabled = !isDirEmpty,
                onClicked = {
                    EventBus.getDefault().post(Events.enterChatEvent(true))
                    onNextClicked() // Enter the chatscreen
                }
            )
        }
    }
}

@Composable
fun HeaderProgressDialog(
    isEnabled: Boolean,
    onClicked: () -> Unit
) {
    AlertDialog(
        onDismissRequest = {},
        text = {
            Text(text = "Preparing inference resources, please wait until the start button is ready.")
        },
        confirmButton = {
            Button(
                enabled = isEnabled,  // Control the enabled state of the button
                onClick = onClicked
            ) {
                Text(text = "Start")
            }
        }
    )
}


@Preview
@Composable
fun SelectOptionPreview(){
    Distributed_inference_demoTheme {
//        ModelSelectionScreen(
//            viewModel = InferenceViewModel(),
//            options = listOf("Model 1", "Model 2", "Model 3", "Model 4"),
//            onCancelClicked = {},
//            onNextClicked = {},
//            modifier = Modifier.fillMaxHeight()
//        )
    }
}