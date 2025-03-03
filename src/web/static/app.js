let socket;
let mediaRecorder;
let audioChunks = [];
let silenceTimer;
let recordingInProgress = false; // Flag to track recording status
const VAD_THRESHOLD = 500; // 2 seconds of silence
let currentStream = null; // Store the active media stream

// Global conversation history object
let conversationHistory = {};
let currentSessionId = null;
let sessionCounter = 0; // Counter to create unique session names

// Initialize SocketIO connection
function initializeSocket() {
    socket = io.connect('/');

    socket.on('connect', () => {
        console.log('Connected to server');
        document.getElementById('status').textContent = 'Status: Connected to server';
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        document.getElementById('status').textContent = 'Status: Disconnected from server';
    });

    socket.on('transcription', (data) => {
        console.log('Transcription received from server');
        const transcription = data.transcription;
        addChatMessage('User', transcription, 'user-message');
        toggleRecordingState(false); // Stop recording after receiving transcription
    });

    socket.on('llm_response', (data) => {
        console.log('LLM response received from server');
        const responseText = data.response;
        addChatMessage('Bot', responseText, 'bot-message');
    });

    socket.on('audio', (data) => {
        console.log('Audio data received from server');
        const response = JSON.parse(data);
        const audioData = response.audio;

        const audioBlob = new Blob([Uint8Array.from(atob(audioData), c => c.charCodeAt(0))], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = document.getElementById('playback');
        audio.src = audioUrl;
        audio.play();
    });

    socket.on('error', (data) => {
        console.error('Error:', data.message);
        toggleRecordingState(false); // Stop recording on error
    });
}

// Toggle Recording State
function toggleRecordingState(isRecording) {
    const recordButton = document.getElementById('record-button');
    recordingInProgress = isRecording;

    if (isRecording) {
        recordButton.style.backgroundColor = '#349057'; // Green for active recording
        recordButton.textContent = '●'; // Active recording icon
        startRecording();
    } else {
        recordButton.style.backgroundColor = '#d63031'; // Gray for inactive
        recordButton.textContent = '●'; // Inactive recording icon
        stopRecording();
    }
}

// Start recording
function startRecording() {
    if (!currentSessionId) {
        // Create a new session automatically if none exists
        startNewSession();
    }

    // Reuse the current stream if available
    if (currentStream) {
        startMediaRecorder(currentStream);
    } else {
        navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
            currentStream = stream; // Save the stream for reuse
            startMediaRecorder(stream);
        }).catch(error => {
            console.error('Error accessing media devices.', error);
        });
    }
}

function startMediaRecorder(stream) {
    audioChunks = []; // Reset the audio chunks
    mediaRecorder = new MediaRecorder(stream);

    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(2048, 1, 1);

    processor.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0);
        const inputDataLength = inputData.length;
        let total = 0;

        for (let i = 0; i < inputDataLength; i++) {
            total += Math.abs(inputData[i]);
        }

        const average = total / inputDataLength;

        if (average < 0.01) { // Silence detection threshold
            if (!silenceTimer) {
                silenceTimer = setTimeout(() => {
                    mediaRecorder.stop(); // Stop the recording after silence
                }, VAD_THRESHOLD);
            }
        } else {
            clearTimeout(silenceTimer);
            silenceTimer = null;
        }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);

    mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.readAsArrayBuffer(audioBlob);
        reader.onloadend = () => {
            console.log('Sending audio to server');
            socket.emit('audio', reader.result);

            // Restart recording after sending audio to the server if the session is still active
            if (recordingInProgress) {
                startRecording(); // Restart recording
            }
        };
    };

    mediaRecorder.start();
    document.getElementById('status').textContent = 'Status: Recording...';
}

// Stop recording
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop(); // Stop the media recorder
    }
    document.getElementById('status').textContent = 'Status: Idle';
    recordingInProgress = false; // Reset recording flag
}

// Start a new session
function startNewSession() {
    sessionCounter++;
    const sessionId = `session-${sessionCounter}`;
    conversationHistory[sessionId] = []; // Initialize an empty array for the session
    currentSessionId = sessionId;
    updateSessionList(); // Update the sidebar with the new session
    clearChatHistory(); // Clear the chat history for the new session
}

// Switch between sessions
function switchSession(sessionId) {
    currentSessionId = sessionId;
    displayChatHistory(sessionId);
}

// Update the session list in the sidebar
function updateSessionList() {
    const sessionListContainer = document.getElementById('session-list');
    sessionListContainer.innerHTML = ''; // Clear the list

    for (const sessionId in conversationHistory) {
        const sessionButton = document.createElement('button');
        sessionButton.textContent = `Conversation ${sessionId.split('-')[1]}`;
        sessionButton.dataset.sessionId = sessionId;
        sessionButton.addEventListener('click', () => switchSession(sessionId));
        sessionListContainer.appendChild(sessionButton);
    }
}

// Display chat history for a specific session
function displayChatHistory(sessionId) {
    clearChatHistory(); // Clear existing chat history
    const sessionHistory = conversationHistory[sessionId];
    sessionHistory.forEach(message => {
        addChatMessage(message.sender, message.text, message.className);
    });
}

// Clear the chat history display
function clearChatHistory() {
    const responseContainer = document.getElementById('response-container');
    responseContainer.innerHTML = ''; // Clear the chat messages
}

// Add chat message to the UI and conversation history
function addChatMessage(sender, message, className) {
    if (!currentSessionId) {
        alert("Create a new conversation first!");
        return;
    }

    // Add message to the UI
    const responseContainer = document.getElementById('response-container');
    const newMessage = document.createElement('div');
    newMessage.textContent = `${sender}: ${message}`;
    newMessage.className = `chat-message ${className}`;
    responseContainer.appendChild(newMessage);
    responseContainer.scrollTop = responseContainer.scrollHeight;

    // Save to conversation history
    conversationHistory[currentSessionId].push({ sender, text: message, className });
}

// Toggle the sidebar when minimize button is clicked
let sidebar = document.getElementById('sidebar');
let main = document.getElementById('main');
let minimizeBtn = document.getElementById('minimize-btn');

minimizeBtn.addEventListener('click', () => {
    sidebar.classList.toggle('minimized');
    main.classList.toggle('expanded');

    // Change the minimize button icon based on the sidebar state
    if (sidebar.classList.contains('minimized')) {
        minimizeBtn.textContent = '⮚'; // Change icon to show expand
    } else {
        minimizeBtn.textContent = '⮘'; // Change icon to show collapse
    }
});

// Add event listener to the record button
document.getElementById('record-button').addEventListener('click', () => {
    toggleRecordingState(!recordingInProgress);
});

// Add event listener to create a new session manually
document.getElementById('add-session-btn').addEventListener('click', startNewSession);

// Initialize the socket connection on page load
initializeSocket();
