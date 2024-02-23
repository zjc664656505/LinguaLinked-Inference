package com.example.distribute_ui.data

import com.example.distribute_ui.ui.ChatUiState
import com.example.distribute_ui.ui.Message

val initialMessages = mutableListOf(
    Message(
        "Robot",
        "Test",
        "03:07 pm",
        null
    ),
    Message(
        "Me",
        "Test Reply",
        "03:07 pm",
        null
    )
)

val exampleUiState = ChatUiState(
    initialMessages = initialMessages
)

val exampleModelName = listOf(
        "bloom560m",
        "bloom3b",
        "bloom1b1",
        "bloom1b7",
        "opt125m",
        "opt350m",
        "opt1b3"
)

val modelMap: HashMap<String, String> = hashMapOf(
    "Bloom" to "bloom560m"
)
