package com.example.distribute_ui.ui

import androidx.compose.runtime.Immutable
import androidx.compose.runtime.toMutableStateList

class ChatUiState(
    initialMessages: MutableList<Messaging> = mutableListOf()
    ) {
    val _chatHistory: MutableList<Messaging> = initialMessages.toMutableStateList()
    val chatHistory = _chatHistory
    var modelName: String = ""
}

@Immutable
data class Messaging(
    val author: String,
    val content: String,
    val timestamp: String,
    val image: Int? = null
)