package com.example.distribute_ui.ui

import android.util.Log
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.exclude
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.ScaffoldDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.rememberTopAppBarState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.LastBaseline
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.distribute_ui.R
import com.example.distribute_ui.TAG
import com.example.distribute_ui.data.initialMessages
import com.example.distribute_ui.ui.components.UserInput
import com.example.distribute_ui.ui.theme.Distributed_inference_demoTheme
import kotlinx.coroutines.launch

const val ConversationTestTag = "ConversationTestTag"

/**
 * @param uiState [ChatUiState] that contains inference result messages to display
 * @param navigateToModelSelection User action when navigation to previous model selection
 * @param modifier [Modifier] to apply to this layout node
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    viewModel: InferenceViewModel,
    modifier: Modifier = Modifier
) {
    val authorMe = stringResource(R.string.author_me)

    // need to change to real time
    val timeNow = "03:07 pm"

    val scrollState = rememberLazyListState()
    val topBarState = rememberTopAppBarState()
    val scrollBehavior = TopAppBarDefaults.pinnedScrollBehavior(topBarState)
    val scope = rememberCoroutineScope()

    val uiState = viewModel.uiState.collectAsState()

    Scaffold(
        topBar = {
            ChatBar(
                modifier = modifier,
                title = {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        // Model name
                        Text(
                            text = uiState.value.modelName,
                            style = MaterialTheme.typography.titleMedium
                        )
                        // Number of Devices
                        Text(
                            text = "Connected Devices: ${uiState.value.connectedDevices}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            )
        },
        contentWindowInsets = ScaffoldDefaults
            .contentWindowInsets
            .exclude(WindowInsets.navigationBars)
            .exclude(WindowInsets.ime),
//        modifier = modifier.nestedScroll(scrollBehavior.nestedScrollConnection)
        modifier = modifier
    ) { paddingValues ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            Messages(
                messages = uiState.value.messages,
//                messages = initialMessages,
                modifier = Modifier.weight(1f),
                scrollState = scrollState
            )
            UserInput(
                onClicked = {
//                    viewModel.inferenceExecution()
//                    viewModel.testInference()
                },
                onMessageSent = { content ->
                    viewModel.addMessage(
                        Message(authorMe, content, timeNow)
                    )
                    viewModel.inferenceExecution(content)
//                    viewModel.inferenceExecution()

                },
                resetScroll = {
                    scope.launch {
                        scrollState.scrollToItem(0)
                    }
                },
                // let this element handle the padding so that the elevation is shown behind the
                // navigation bar
                modifier = Modifier
                    .navigationBarsPadding()
                    .imePadding()
                    .navigationBarsPadding()
                    .imePadding()
                    .fillMaxWidth()
                    .padding(vertical = 8.dp)

            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatBar(
    modifier: Modifier = Modifier,
    title: @Composable () -> Unit
) {
    CenterAlignedTopAppBar(
        modifier = modifier,
        title = title
    )
}

@Composable
fun Messages(
    messages: List<Message>,
    scrollState: LazyListState,
    modifier: Modifier = Modifier
) {
    val scope = rememberCoroutineScope()
    val authorMe = stringResource(id = R.string.author_me)
    Box(modifier = modifier) {
        LazyColumn(
            reverseLayout = true,
            state = scrollState,
            modifier = Modifier
                .testTag(ConversationTestTag)
                .fillMaxSize()
        ) {
            for (index in messages.indices) {
//                val prevAuthor = messages.getOrNull(index - 1)?.author
//                val nextAuthor = messages.getOrNull(index + 1)?.author
                val content = messages[index]
                Log.d(TAG, "content is $content")
//                val isFirstMessageByAuthor = prevAuthor != content.author
//                val isLastMessageByAuthor = nextAuthor != content.author
//
//                // Hardcode day dividers for simplicity
//                if (index == messages.size - 1) {
//                    item {
//                        DayHeader("20 Aug")
//                    }
//                } else if (index == 2) {
//                    item {
//                        DayHeader("Today")
//                    }
//                }

                item {
                    Message(
                        msg = content,
                        isUserMe = content.author == authorMe,
                    )
                }
            }
        }
    }
}

@Composable
fun Message(
    msg: Message,
    isUserMe: Boolean,
) {
    val borderColor = if (isUserMe) {
        MaterialTheme.colorScheme.primary
    } else {
        MaterialTheme.colorScheme.tertiary
    }

    Column(
        modifier = Modifier.padding(horizontal = 8.dp)
    ) {
        Row(modifier = Modifier.semantics(mergeDescendants = true) {}) {
            Text(
                text = msg.author,
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier
                    .alignBy(LastBaseline)
//                .paddingFrom(LastBaseline, after = 8.dp) // Space to 1st bubble
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = msg.timestamp,
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.alignBy(LastBaseline),
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Text(
            text = msg.content,
            style = MaterialTheme.typography.bodySmall
        )
    }

//    val spaceBetweenAuthors = Modifier
//    Row(modifier = spaceBetweenAuthors) {
//        Spacer(modifier = Modifier.width(74.dp))
//        AuthorAndTextMessage(
//            msg = msg,
//            isUserMe = isUserMe,
//            modifier = Modifier
//                .padding(end = 16.dp)
//                .weight(1f)
//        )
//    }
}

@Composable
fun AuthorAndTextMessage(
    msg: Message,
    isUserMe: Boolean,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        AuthorNameTimestamp(msg)
//        ChatItemBubble(msg, isUserMe)
        Spacer(modifier = Modifier.height(8.dp))
    }
}

@Composable
private fun AuthorNameTimestamp(msg: Message) {
    // Combine author and timestamp for a11y.
    Row(modifier = Modifier.semantics(mergeDescendants = true) {}) {
        Text(
            text = msg.author,
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier
                .alignBy(LastBaseline)
//                .paddingFrom(LastBaseline, after = 8.dp) // Space to 1st bubble
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = msg.timestamp,
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.alignBy(LastBaseline),
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

private val ChatBubbleShape = RoundedCornerShape(4.dp, 20.dp, 20.dp, 20.dp)

@Composable
fun DayHeader(dayString: String) {
    Row(
        modifier = Modifier
            .padding(vertical = 8.dp, horizontal = 16.dp)
            .height(16.dp)
    ) {
        DayHeaderLine()
        Text(
            text = dayString,
            modifier = Modifier.padding(horizontal = 16.dp),
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        DayHeaderLine()
    }
}

@Composable
private fun RowScope.DayHeaderLine() {
    Divider(
        modifier = Modifier
            .weight(1f)
            .align(Alignment.CenterVertically),
        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.12f)
    )
}

@Composable
fun ChatItemBubble(
    message: Message,
    isUserMe: Boolean,
) {

    val backgroundBubbleColor = if (isUserMe) {
        MaterialTheme.colorScheme.primary
    } else {
        MaterialTheme.colorScheme.surfaceVariant
    }

    Column {
        Surface(
            color = backgroundBubbleColor,
            shape = ChatBubbleShape
        ) {

        }

        message.image?.let {
            Spacer(modifier = Modifier.height(4.dp))
            Surface(
                color = backgroundBubbleColor,
                shape = ChatBubbleShape
            ) {
                Image(
                    painter = painterResource(it),
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.size(160.dp),
                    contentDescription = stringResource(id = R.string.attached_image)
                )
            }
        }
    }
}


@Preview
@Composable
fun ConversationPreview() {
    Distributed_inference_demoTheme {
//        ChatScreen(viewModel = InferenceViewModel())
    }
}

@Preview
@Composable
fun MessagesPreview() {
    Distributed_inference_demoTheme {
        Messages(messages = initialMessages, scrollState = rememberLazyListState())
    }
}