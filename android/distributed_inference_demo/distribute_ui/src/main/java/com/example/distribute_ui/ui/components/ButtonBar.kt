package com.example.distribute_ui.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.selection.selectable
import androidx.compose.material3.Button
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.dimensionResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.distribute_ui.R
import com.example.distribute_ui.ui.ModelSelectionScreen
import com.example.distribute_ui.ui.theme.Distributed_inference_demoTheme

@Composable
fun ButtonBar(
    modifier: Modifier = Modifier,
    onNextClicked: () -> Unit,
    onCancelClicked: () -> Unit,
    selectedValue: Boolean

) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(dimensionResource(R.dimen.padding_medium)),
        horizontalArrangement = Arrangement.spacedBy(dimensionResource(R.dimen.padding_medium)),
        verticalAlignment = Alignment.Bottom

    ) {
        OutlinedButton(
            modifier = Modifier.weight(1f),
            onClick = onCancelClicked
        ) {
            Text(stringResource(R.string.cancel))
        }
        Button(
            modifier = Modifier.weight(1f),
            // the button is enabled when the user makes a selection
            enabled = selectedValue,
            onClick = onNextClicked
        ) {
            Text(stringResource(R.string.next))
        }
    }
}

@Preview
@Composable
fun SelectOptionPreview(){
    Distributed_inference_demoTheme {
        ButtonBar(
            onNextClicked = { /*TODO*/ },
            onCancelClicked = { /*TODO*/ },
            selectedValue = true)
    }
}