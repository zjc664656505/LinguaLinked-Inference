package com.example.distribute_ui.ui
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.LastBaseline
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.distribute_ui.BackgroundService
import com.example.distribute_ui.DataRepository
import com.example.distribute_ui.R
import com.example.distribute_ui.TAG
import com.example.distribute_ui.data.initialMessages
import com.example.distribute_ui.ui.components.UserInput
import com.example.distribute_ui.ui.theme.Distributed_inference_demoTheme
import kotlinx.coroutines.launch
import org.greenrobot.eventbus.EventBus
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import com.example.distribute_ui.Events


const val ConversationTestTag = "ConversationTestTag"

/**
 * @param uiState [ChatUiState] that contains inference result messages to display
 * @param navigateToModelSelection User action when navigation to previous model selection
 * @param modifier [Modifier] to apply to this layout node
 */
@RequiresApi(Build.VERSION_CODES.O)
@OptIn(ExperimentalMaterial3Api::class)



@Composable
fun ChatScreen(
    viewModel: InferenceViewModel,
    modifier: Modifier = Modifier
) {
    val authorMe = stringResource(R.string.author_me)

    val currentTimeMillis = System.currentTimeMillis()
    // Convert milliseconds since the epoch to an Instant
    val instant = Instant.ofEpochMilli(currentTimeMillis)
    // Format the Instant as a String
    val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss").withZone(ZoneId.systemDefault())
    val formatted = formatter.format(instant)
    val timeNow = formatted
    val context = LocalContext.current
    val scrollState = rememberLazyListState()
    val topBarState = rememberTopAppBarState()
    val scrollBehavior = TopAppBarDefaults.pinnedScrollBehavior(topBarState)
    val scope = rememberCoroutineScope()
    val uiState = viewModel.uiState.collectAsState()

    // observe model response
    val decodedString by DataRepository.decodingStringLiveData.observeAsState("")

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
                        // Show greeting text
                        Text(
                            text = "Welcome to LinguaLinked Chat.",
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
                decodedString = decodedString,
                modifier = Modifier.weight(1f),
                scrollState = scrollState
            )
            UserInput(
                onClicked = {
                },
                onMessageSent = { content ->
                    viewModel.addMessage(
                        Message(authorMe, content, timeNow)
                    )
                    EventBus.getDefault().post(Events.messageSentEvent(true, content))
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
    decodedString: String, // Correct parameter name to match Kotlin conventions
    scrollState: LazyListState,
    modifier: Modifier = Modifier
) {
    Box(modifier = modifier) {
        LazyColumn(
            state = scrollState,
            modifier = Modifier.fillMaxSize()
        ) {
            items(messages.size) { index ->
                val message = messages[index]
                Message(
                    msg = message,
                    isUserMe = message.author == "Me" // Assuming the author "Me" represents the user
                )
            }

            // Check if decodedString is not empty and display it
            if (decodedString.isNotEmpty()) {
                item {
                    // Display decodedString in a single message box
                    // Assuming the model is represented as "Model"
                    Message(
                        msg = Message(author = "LinguaLinked", content = decodedString, timestamp = "Now"),
                        isUserMe = false // Model messages are not from the user
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
            style = MaterialTheme.typography.bodyMedium
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
        Messages(messages = initialMessages, decodedString="example model response", scrollState = rememberLazyListState())
    }
}