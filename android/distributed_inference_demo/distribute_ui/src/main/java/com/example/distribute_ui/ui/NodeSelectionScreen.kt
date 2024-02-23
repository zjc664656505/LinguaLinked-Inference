package com.example.distribute_ui.ui

import android.content.Intent
import android.util.Log
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.dimensionResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.distribute_ui.R
import com.example.distribute_ui.TAG
import com.example.distribute_ui.service.InferenceService
import com.example.distribute_ui.ui.components.ButtonBar
import com.example.distribute_ui.ui.theme.Distributed_inference_demoTheme

@Composable
fun NodeSelectionScreen(
    viewModel: InferenceViewModel,
    onHeaderClicked: () -> Unit,
    onWorkerClicked: () -> Unit,
    onCancelClicked: () -> Unit,
    onNextClicked: (id: Int) -> Unit,
    onBackendStarted: () -> Unit,
    onMonitorStarted: () -> Unit,
    modifier: Modifier = Modifier
) {
    val mediumPadding = dimensionResource(R.dimen.padding_medium)
    val selectedValue = remember { mutableStateOf(false) }
    // 0 for worker, 1 for header
    val selectionNode = remember { mutableStateOf(0) }
    val nextClickedState = remember { mutableStateOf(false) }

    val buttonEnable = remember {
        mutableStateOf(false)
    }

//    val context = LocalContext.current

    Column(
        modifier = modifier
            .fillMaxHeight()
            .verticalScroll(rememberScrollState())
            .padding(mediumPadding),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        NodeSelectionLayout(
            modifier = Modifier
                .fillMaxWidth()
                .wrapContentHeight()
                .padding(mediumPadding),
            onHeaderClicked = {
                selectedValue.value = true
                selectionNode.value = 1
                onHeaderClicked()
            },
            onWorkerClicked = {
                selectedValue.value = true
                selectionNode.value = 0
                onWorkerClicked()
            },
            selectedValue = selectedValue.value
        )
    }
    ButtonBar(
        modifier = modifier,
        onNextClicked = {
            nextClickedState.value = true
//            if (selectionNode.value == 1) {
//                onNextClicked()
//            }
            onNextClicked(selectionNode.value)
//            onBackendStarted()
//            onMonitorStarted()
        },
        onCancelClicked = {
            selectedValue.value = false
            onCancelClicked()
        },
        selectedValue = selectedValue.value)
    if (nextClickedState.value && selectionNode.value == 0) {
        onBackendStarted()
//        viewModel.updateDeviceInfo()
//        viewModel.inferencePrepare()
        WorkerProgressDialog(
            enable = buttonEnable.value,
            onClicked = {

                // test monitor service
//                onMonitorStarted()

//                viewModel.inferencePrepare()
//                buttonEnable.value = false
//                viewModel.startService()
//                Log.d(TAG, "enter onclick in dialog")
//                viewModel.inferenceExecution()
//                viewModel.runInfer()
//                val serviceIntent = Intent(context, InferenceService::class.java)
//                context.startService(serviceIntent)
            }
        )
    }
}

@Composable
fun NodeSelectionLayout(
    modifier: Modifier = Modifier,
    onHeaderClicked: () -> Unit,
    onWorkerClicked : () -> Unit,
    selectedValue: Boolean,
) {
    val mediumPadding = dimensionResource(R.dimen.padding_medium)
    val masterEnable = remember { mutableStateOf(false) }
    var masterButtonColor = if (masterEnable.value) ButtonDefaults.buttonColors() else ButtonDefaults.outlinedButtonColors()
    if (!selectedValue) {
        masterButtonColor = ButtonDefaults.outlinedButtonColors()
    }
    val workerEnable = remember { mutableStateOf(false) }
    var workerButtonColor = if (workerEnable.value) ButtonDefaults.buttonColors() else ButtonDefaults.outlinedButtonColors()
    if (!selectedValue) {
        workerButtonColor = ButtonDefaults.outlinedButtonColors()
    }
    Card(
        modifier = modifier,
        elevation = CardDefaults.cardElevation(defaultElevation = 5.dp)
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(mediumPadding),
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(mediumPadding)
        ) {

            Text(
                text = stringResource(R.string.select_mode_str),
                textAlign = TextAlign.Center,
                style = MaterialTheme.typography.bodyLarge
            )
            OutlinedButton(
                onClick = {
                    workerEnable.value = false
                    masterEnable.value = true
                    onHeaderClicked()
                },
                modifier = Modifier.fillMaxHeight(),
                colors = masterButtonColor
            ) {
                Text(
                    text = stringResource(R.string.button_master),
                    fontSize = 14.sp
                )
            }
            OutlinedButton(
                onClick = {
                    masterEnable.value = false
                    workerEnable.value = true
                    onWorkerClicked()
                },
                modifier = Modifier.fillMaxHeight(),
                colors = workerButtonColor
            ) {
                Text(
                    text = stringResource(R.string.button_worker),
                    fontSize = 14.sp
                )
            }
        }
    }
}


@Composable
fun WorkerProgressDialog(
    enable: Boolean,
    onClicked: () ->  Unit
) {
    AlertDialog(
        onDismissRequest = {},
        text = {
            Text(text = "The system starts working, please wait.")
        },
        confirmButton = {
//            Button(
//                enabled = enable,
//                onClick = onClicked
//            ) {
//                Text(text = "start")
//            }
        }
    )
}

@Preview(showBackground = true)
@Composable
fun NodeSelectionScreenPreview() {
//    Distributed_inference_demoTheme {
//        NodeSelectionScreen(
//            viewModel = InferenceViewModel(),
//            onHeaderClicked = {},
//            onWorkerClicked = {},
//            onCancelClicked = {},
//            onNextClicked = {},
//            modifier = Modifier.fillMaxHeight(),
//        )
//    }
}
