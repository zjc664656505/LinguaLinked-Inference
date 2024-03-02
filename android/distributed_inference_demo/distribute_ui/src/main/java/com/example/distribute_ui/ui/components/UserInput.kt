package com.example.distribute_ui.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.distribute_ui.R

@Preview
@Composable
fun UserInputPreview() {
    UserInput(
        onClicked = {},
        onMessageSent = {},
        modifier = Modifier.fillMaxWidth()
    )
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalComposeUiApi::class)
@Composable
fun UserInput(
    onClicked: () -> Unit,
    onMessageSent: (String) -> Unit,
    modifier: Modifier = Modifier,
    resetScroll: () -> Unit = {},
) {

    var textState by rememberSaveable(stateSaver = TextFieldValue.Saver) {
        mutableStateOf(TextFieldValue())
    }
    val keyboardController = LocalSoftwareKeyboardController.current
    // Used to decide if the keyboard should be shown
//    var textFieldFocusState by remember { mutableStateOf(false) }

    Surface(tonalElevation = 2.dp) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Center,
            modifier = modifier
                .height(72.dp)
//                .wrapContentHeight()
                .fillMaxWidth()
                .padding(horizontal = 4.dp)
        ) {
            TextField(
                value = textState,
                onValueChange = { textState = it },
                label = { Text("Enter Message")},
                keyboardOptions = KeyboardOptions(
                    imeAction = ImeAction.Send
                ),
                keyboardActions = KeyboardActions(
                    onSend = {
                        if (textState.text.isNotBlank()) {
                            keyboardController?.hide()
                            onMessageSent(textState.text)
                            onClicked()

                            // Reset text field and close keyboard
                            textState = TextFieldValue()
                            resetScroll()
                        }
                    }
                ),
                maxLines = 2,
                modifier = Modifier.weight(0.9f)
            )
            Spacer(modifier = Modifier.size(8.dp))

            // Send button
            val disabledContentColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
            val buttonColors = ButtonDefaults.buttonColors(
                disabledContainerColor = Color.Transparent,
                disabledContentColor = disabledContentColor
            )
            val border = if (!textState.text.isNotBlank()) {
                BorderStroke(
                    width = 1.dp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
                )
            } else {
                null
            }
            Button(
//                enabled = textState.text.isNotBlank(),
                enabled = true,
                onClick = {
                    if (textState.text.isNotBlank()) {
                        keyboardController?.hide()

                        onMessageSent(textState.text)
                        onClicked()
                        // Reset text field and close keyboard
                        textState = TextFieldValue()
                        resetScroll()
                    }
                    // Move scroll to bottom
//                    resetScroll()
                },
                colors = buttonColors,
                border = border,
                contentPadding = PaddingValues(0.dp),
                modifier = Modifier.weight(0.1f)
            ) {
                Text(
                    stringResource(id = R.string.send),
                    modifier = Modifier
//                        .padding(horizontal = 16.dp)
                        .fillMaxWidth(),
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}