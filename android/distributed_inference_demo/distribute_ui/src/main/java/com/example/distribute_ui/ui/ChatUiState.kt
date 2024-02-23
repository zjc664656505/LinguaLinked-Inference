package com.example.distribute_ui.ui

import androidx.compose.runtime.Immutable
import androidx.compose.runtime.toMutableStateList

class ChatUiState(
    initialMessages: MutableList<Message> = mutableListOf()
    ) {
    val _messages: MutableList<Message> = initialMessages.toMutableStateList()
    val messages: List<Message> = _messages
    var modelName: String = ""
    var connectedDevices: Int = 0

//    fun addMessage(msg: Message) {
//        _messages.add(0, msg) // Add to the beginning of the list
//    }
}

@Immutable
data class Message(
    val author: String,
    val content: String,
    val timestamp: String,
    val image: Int? = null
)