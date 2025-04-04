// Global variable to hold the upload queue (not used with server-side approach)
// let uploadQueue = [];

// Initialize the application
document.addEventListener("DOMContentLoaded", function() {
    // Load documents list if we're on the documents tab
    if (document.getElementById("documents-tab").classList.contains("active")) {
        loadDocuments();
        loadUploadQueue();
    }
    
    // Load current LLM endpoint if we're on the admin tab
    if (document.getElementById("admin-tab").classList.contains("active")) {
        loadCurrentEndpoint();
    }
    
    // Load current endpoint for the quick update form
    loadQuickEndpoint();
    
    // Add event listener for Enter key on query input
    document.getElementById("queryInput").addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            askQuestion();
        }
    });
    
    // Add event listener for Enter key on quick endpoint input
    document.getElementById("quickEndpointInput").addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            updateQuickEndpoint();
        }
    });
});

/**
 * Tab Navigation
 */
function openTab(tabName) {
    // Hide all tab contents
    let tabContents = document.getElementsByClassName("tab-content");
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove("active");
    }
    
    // Remove active class from all tabs
    let tabs = document.getElementsByClassName("tab");
    for (let i = 0; i < tabs.length; i++) {
        tabs[i].classList.remove("active");
    }
    
    // Show the selected tab content and mark the tab as active
    document.getElementById(tabName).classList.add("active");
    
    // Find and activate the corresponding tab
    let allTabs = document.getElementsByClassName("tab");
    for (let i = 0; i < allTabs.length; i++) {
        if (allTabs[i].getAttribute("onclick").includes(tabName)) {
            allTabs[i].classList.add("active");
        }
    }
    
    // Load content based on the active tab
    if (tabName === "documents-tab") {
        loadDocuments();
        loadUploadQueue();
    } else if (tabName === "admin-tab") {
        loadCurrentEndpoint();
    }
}

/**
 * Document Management Functions
 */
async function loadDocuments() {
    try {
        const response = await fetch('/documents/');
        const result = await response.json();
        
        const documentList = document.getElementById('documentList');
        
        if (result.documents && result.documents.length > 0) {
            let html = '';
            for (const doc of result.documents) {
                // Format last modified time if available
                const lastModified = doc.last_modified ? 
                    `<span>Modified: ${doc.last_modified}</span>` : '';
                
                // Format chunks info with more detail
                const chunksInfo = doc.chunks > 1 ? 
                    `<span class="document-chunks" title="Document is split into smaller chunks for better retrieval">
                        ${doc.chunks} chunks
                     </span>` : 
                    `<span class="document-chunks" style="background-color: #E0E0E0;">Single chunk</span>`;
                
                html += `
                    <div class="document-item">
                        <div class="document-header">
                            <div class="document-info">
                                <strong>${doc.filename}</strong>
                                <span class="document-type">${doc.type}</span>
                                ${chunksInfo}
                            </div>
                            <div>${doc.size}</div>
                        </div>
                        <div class="document-meta">
                            ${lastModified}
                            <span>Indexed in vector store</span>
                        </div>
                    </div>
                `;
            }
            documentList.innerHTML = html;
        } else {
            documentList.innerHTML = '<div style="padding: 15px; background: #f5f5f5; border-radius: 4px; text-align: center;">No documents have been indexed in the vector store yet.<br>Add files to the queue and index them to see them here.</div>';
        }
    } catch (error) {
        console.error('Error loading documents:', error);
        documentList.innerHTML = '<div style="padding: 15px; background: #ffebee; border-radius: 4px; color: #c62828; text-align: center;">Failed to load vector store documents. Please try refreshing.</div>';
    }
}

function addFilesToUploadQueue() {
    let fileInput = document.getElementById("fileInput");
    let files = fileInput.files;
    
    if (files.length === 0) {
        alert("Please select at least one file!");
        return;
    }
    
    // Create form to upload files to server
    let formData = new FormData();
    for (let file of files) {
        formData.append("files", file);
    }
    
    // Show loading message
    const queueElement = document.getElementById("uploadQueue");
    queueElement.innerHTML = '<div style="text-align: center; padding: 20px;">Uploading files to queue...</div>';
    
    // Upload files to server (but don't index yet)
    fetch("/upload-only/", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(result => {
        // Show confirmation
        if (result.files && result.files.length > 0) {
            alert(`Added ${result.files.length} file${result.files.length > 1 ? 's' : ''} to the upload queue. Use the 'Index Queued Files' button when ready to add them to the vector store.`);
        } else {
            alert("No new files were added.");
        }
        
        // Refresh the queue
        loadUploadQueue();
        
        // Clear the file input
        fileInput.value = "";
    })
    .catch(error => {
        console.error("Error uploading files:", error);
        alert("Error uploading files. Please try again.");
        loadUploadQueue();
    });
}

async function loadUploadQueue() {
    const queueElement = document.getElementById("uploadQueue");
    const indexButton = document.getElementById("indexButton");
    
    queueElement.innerHTML = '<div style="text-align: center; padding: 10px;">Loading queue...</div>';
    
    try {
        const response = await fetch('/upload-queue/');
        const result = await response.json();
        
        if (result.files && result.files.length > 0) {
            let html = '';
            result.files.forEach((file, index) => {
                html += `
                    <div class="document-item">
                        <div class="document-header">
                            <div class="document-info">
                                <strong>${file.filename}</strong>
                                <span class="document-type">${file.type || 'Unknown'}</span>
                            </div>
                            <div>${file.size}</div>
                        </div>
                        <div class="document-meta">
                            <span>Modified: ${file.last_modified}</span>
                            <button onclick="removeFileFromQueue('${file.filename}')" style="width: auto; padding: 2px 8px; background-color: #F44336; margin-top: 0;">Remove</button>
                        </div>
                    </div>
                `;
            });
            queueElement.innerHTML = html;
            indexButton.disabled = false;
            indexButton.style.backgroundColor = "#1976D2";
            indexButton.textContent = `Index Queued Files (${result.files.length})`;
        } else {
            queueElement.innerHTML = "<p>No files in queue</p>";
            indexButton.disabled = true;
            indexButton.style.backgroundColor = "#757575";
            indexButton.textContent = "Index Queued Files (0)";
        }
    } catch (error) {
        console.error("Error loading upload queue:", error);
        queueElement.innerHTML = '<div style="padding: 15px; background: #ffebee; border-radius: 4px; color: #c62828; text-align: center;">Failed to load upload queue. Please try refreshing.</div>';
    }
}

async function removeFileFromQueue(filename) {
    if (!confirm(`Are you sure you want to remove "${filename}" from the queue?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/remove-from-queue/?filename=${encodeURIComponent(filename)}`, {
            method: "DELETE"
        });
        
        if (response.ok) {
            // Refresh the queue
            loadUploadQueue();
        } else {
            alert("Error removing file from queue.");
        }
    } catch (error) {
        console.error("Error removing file:", error);
        alert("Error removing file from queue.");
    }
}

async function indexQueuedDocuments() {
    try {
        const response = await fetch('/upload-queue/');
        const result = await response.json();
        
        if (!result.files || result.files.length === 0) {
            alert("No files in queue to index!");
            return;
        }
        
        // Get list of filenames to index
        const filenames = result.files.map(file => file.filename);
        
        // Send request to index files
        const indexResponse = await fetch("/index-files/", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ filenames: filenames })
        });
        
        const indexResult = await indexResponse.json();
        
        alert(`${indexResult.message} Index time: ${indexResult.process_time}`);
        
        // Refresh both document lists
        loadUploadQueue();
        loadDocuments();
        
    } catch (error) {
        console.error("Error indexing files:", error);
        alert("Error indexing files. Please try again.");
    }
}

/**
 * Chat Functions
 */
async function askQuestion() {
    let query = document.getElementById("queryInput").value;
    if (!query) {
        alert("Please enter a question!");
        return;
    }

    // Show loading indicator
    let responseBox = document.getElementById("responseBox");
    responseBox.style.display = "block";
    responseBox.innerHTML = `<b>Question:</b> ${query}<br><br><b>Answer:</b> Loading...`;
    
    // Hide timing and context initially
    document.getElementById("timingBox").style.display = "none";
    document.getElementById("contextContainer").style.display = "none";

    try {
        let response = await fetch(`/ask/?query=${encodeURIComponent(query)}`);
        let result = await response.json();

        // Display question and answer
        responseBox.innerHTML = `<b>Question:</b> ${query}<br><br><b>Answer:</b> ${result.answer}`;
        
        // Display timing information
        let timingBox = document.getElementById("timingBox");
        timingBox.style.display = "block";
        timingBox.innerHTML = `
            <div class="timing-item"><b>Retrieval Time:</b> ${result.timing.retrieval_time}</div>
            <div class="timing-item"><b>LLM Time:</b> ${result.timing.llm_time}</div>
            <div class="timing-item"><b>Total Time:</b> ${result.timing.total_time}</div>
        `;
        
        // Clear the input field after asking
        document.getElementById("queryInput").value = "";
        
        // Handle context separately
        let contextBox = document.getElementById("contextBox");
        let contextContainer = document.getElementById("contextContainer");
        let toggleButton = document.getElementById("toggleContext");
        
        // Reset context display
        contextBox.style.display = "none";
        toggleButton.textContent = "Show Context";
        
        // Format and store context
        if (result.context && result.context.length > 0) {
            contextContainer.style.display = "block";
            contextBox.innerHTML = "<b>Context Used:</b><br>" + result.context.map(ctx => 
                `<div class="context-item">${ctx.source}: ${ctx.content}</div>`
            ).join("");
        } else {
            contextContainer.style.display = "none";
        }
    } catch (error) {
        responseBox.innerHTML = `<b>Question:</b> ${query}<br><br><b>Error:</b> Failed to get response. Please try again.`;
        console.error("Error fetching answer:", error);
    }
}

function toggleContext() {
    let contextBox = document.getElementById("contextBox");
    let toggleButton = document.getElementById("toggleContext");
    
    if (contextBox.style.display === "none") {
        contextBox.style.display = "block";
        toggleButton.textContent = "Hide Context";
    } else {
        contextBox.style.display = "none";
        toggleButton.textContent = "Show Context";
    }
}

/**
 * LLM Endpoint Management Functions
 */
async function loadCurrentEndpoint() {
    let endpointDiv = document.getElementById("currentEndpoint");
    endpointDiv.innerHTML = "Loading current endpoint...";
    
    try {
        let response = await fetch("/admin/llm-endpoint/");
        let result = await response.json();
        
        const endpoint = result.endpoint;
        
        if (!endpoint) {
            endpointDiv.innerHTML = "<strong style='color: #F44336;'>No endpoint set!</strong> Please set an endpoint URL below.";
            return;
        }
        
        endpointDiv.innerHTML = `<strong>Current Endpoint:</strong> ${endpoint}`;
    } catch (error) {
        endpointDiv.innerHTML = "Error loading current endpoint.";
        console.error("Error loading endpoint:", error);
    }
}

async function updateLLMEndpoint() {
    let newEndpoint = document.getElementById("endpointInput").value;
    
    if (!newEndpoint) {
        alert("Please enter a valid endpoint URL");
        return;
    }
    
    // Basic URL validation
    if (!newEndpoint.startsWith("http://") && !newEndpoint.startsWith("https://")) {
        alert("URL must start with http:// or https://");
        return;
    }
    
    let resultDiv = document.getElementById("endpointUpdateResult");
    resultDiv.style.display = "block";
    resultDiv.innerHTML = "Updating endpoint...";
    resultDiv.className = "response";
    
    try {
        // Use a GET request with query params instead of POST with JSON body
        let response = await fetch(`/admin/update-llm-endpoint/?endpoint_url=${encodeURIComponent(newEndpoint)}`);
        let result = await response.json();
        
        if (response.ok) {
            resultDiv.innerHTML = `<strong>Success:</strong> ${result.message}`;
            resultDiv.style.color = "#4CAF50";
            
            // Refresh the current endpoint display
            loadCurrentEndpoint();
            
            // Clear the input field
            document.getElementById("endpointInput").value = "";
        } else {
            resultDiv.innerHTML = `<strong>Error:</strong> ${result.error || "Failed to update endpoint"}`;
            resultDiv.style.color = "#F44336";
        }
    } catch (error) {
        resultDiv.innerHTML = "<strong>Error:</strong> Failed to connect to server";
        resultDiv.style.color = "#F44336";
        console.error("Error updating endpoint:", error);
    }
}

function toggleQuickEndpointForm() {
    const form = document.getElementById("quickEndpointForm");
    const icon = document.getElementById("formToggleIcon");
    
    if (form.classList.contains("active")) {
        form.classList.remove("active");
        icon.textContent = "▲";
    } else {
        form.classList.add("active");
        icon.textContent = "▼";
        loadQuickEndpoint();
    }
}

async function loadQuickEndpoint() {
    const endpointDiv = document.getElementById("quickCurrentEndpoint");
    
    try {
        const response = await fetch("/admin/llm-endpoint/");
        const result = await response.json();
        
        const endpoint = result.endpoint;
        
        if (!endpoint) {
            endpointDiv.innerHTML = "<strong>No endpoint set!</strong> Please set one.";
            endpointDiv.style.color = "#F44336";
            return;
        }
        
        // Truncate if too long
        const displayEndpoint = endpoint.length > 30 ? 
            endpoint.substring(0, 27) + "..." : 
            endpoint;
        
        endpointDiv.innerHTML = `<strong>Current:</strong> ${displayEndpoint}`;
        endpointDiv.title = endpoint; // Show full URL on hover
        endpointDiv.style.color = "";
    } catch (error) {
        endpointDiv.innerHTML = "Error loading endpoint";
        console.error("Error loading endpoint:", error);
    }
}

async function updateQuickEndpoint() {
    const newEndpoint = document.getElementById("quickEndpointInput").value;
    const statusDiv = document.getElementById("quickEndpointStatus");
    
    if (!newEndpoint) {
        statusDiv.textContent = "Please enter a valid URL";
        statusDiv.className = "status error";
        statusDiv.style.display = "block";
        return;
    }
    
    // Basic URL validation
    if (!newEndpoint.startsWith("http://") && !newEndpoint.startsWith("https://")) {
        statusDiv.textContent = "URL must start with http:// or https://";
        statusDiv.className = "status error";
        statusDiv.style.display = "block";
        return;
    }
    
    statusDiv.textContent = "Updating...";
    statusDiv.className = "status";
    statusDiv.style.display = "block";
    
    try {
        const response = await fetch(`/admin/update-llm-endpoint/?endpoint_url=${encodeURIComponent(newEndpoint)}`);
        const result = await response.json();
        
        if (response.ok) {
            statusDiv.textContent = "Updated successfully!";
            statusDiv.className = "status success";
            
            // Refresh both endpoint displays
            loadQuickEndpoint();
            if (document.getElementById("admin-tab").classList.contains("active")) {
                loadCurrentEndpoint();
            }
            
            // Clear the input field
            document.getElementById("quickEndpointInput").value = "";
            
            // Auto-hide the status after 3 seconds
            setTimeout(() => {
                statusDiv.style.display = "none";
            }, 3000);
        } else {
            statusDiv.textContent = result.error || "Update failed";
            statusDiv.className = "status error";
        }
    } catch (error) {
        statusDiv.textContent = "Connection error";
        statusDiv.className = "status error";
        console.error("Error updating endpoint:", error);
    }
} 