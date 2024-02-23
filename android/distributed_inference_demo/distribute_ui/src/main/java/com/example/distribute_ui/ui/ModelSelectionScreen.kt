package com.example.distribute_ui.ui

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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.dimensionResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import com.example.distribute_ui.R
import com.example.distribute_ui.TAG
import com.example.distribute_ui.ui.components.ButtonBar
import com.example.distribute_ui.ui.theme.Distributed_inference_demoTheme

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
//    val ipState by viewModel.IPState.collectAsState()
    val prepareState by viewModel.prepareState.collectAsState()

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
                text = "Number of connected devices: 5",
                textAlign = TextAlign.Left,
                style = MaterialTheme.typography.bodyMedium
            )
        }
        ButtonBar(
            modifier = modifier,
            onNextClicked = {
                nextClickedState.value = true
//                viewModel.selectModel(selectedModel)
//                onNextClicked()
//                viewModel.updateDeviceInfo()
//                viewModel.inferencePrepare()

//                onNextClicked(selectedModel)
            },
            onCancelClicked = {
                nextClickedState.value = false
                selectedValue.value = false
                onCancelClicked()
            },
            selectedValue = selectedValue.value
        )
        if (nextClickedState.value && selectedValue.value) {

            HeaderProgressDialog(
                onClicked = {

//                    viewModel.inferencePrepare()
//                    viewModel.testPrepareState()
                }
            )
            onBackendStarted()
        }
//        if (prepareState) {
//            LaunchedEffect(prepareState) {
//                onNextClicked()
//                viewModel.resetPrepareState()
//                val time = System.currentTimeMillis()
//                Log.d(TAG, "time after onNextClicked in model selection is $time")
//            }
//        }
    }
}

@Composable
fun HeaderProgressDialog(

    onClicked: () -> Unit
) {
    AlertDialog(
        onDismissRequest = {},
        text = {
            Text(text = "The header starts downloading, please wait until the start button is ready.")
        },
        confirmButton = {
            Button(
                enabled = true,
                onClick = onClicked
            ) {
                Text(text = "start")
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