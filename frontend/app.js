// ============================================================================
// Configuration
// ============================================================================
const API_BASE_URL = window.location.origin;

// ============================================================================
// State Management
// ============================================================================
const state = {
    currentTab: 'generate',
    theme: localStorage.getItem('theme') || 'light',
    connected: false,
    ws: null,
    viewMode: 'markdown', // 'markdown' or 'code'
    currentRawResponse: '',
    currentCode: ''
};

// ============================================================================
// Initialization
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initTabs();
    initEventListeners();
    checkHealth();
    loadStats();
    loadDownloadedModels(); // Load models for dropdown
});

// ============================================================================
// Theme Management
// ============================================================================
function initTheme() {
    document.documentElement.setAttribute('data-theme', state.theme);
    updateThemeIcon();
}

function toggleTheme() {
    state.theme = state.theme === 'light' ? 'dark' : 'light';
    localStorage.setItem('theme', state.theme);
    document.documentElement.setAttribute('data-theme', state.theme);
    updateThemeIcon();
}

function updateThemeIcon() {
    const icon = document.querySelector('.theme-icon');
    icon.textContent = state.theme === 'light' ? '🌙' : '☀️';
}

// ============================================================================
// Tab Management
// ============================================================================
function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
}

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === tabName);
    });

    state.currentTab = tabName;

    // Load repositories and stats when switching to index tab
    if (tabName === 'index') {
        loadRepositories();
        loadStats();
    }
}

// ============================================================================
// Event Listeners
// ============================================================================
function initEventListeners() {
    // Theme toggle
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);

    // Generate code
    document.getElementById('generateBtn').addEventListener('click', handleGenerate);
    document.getElementById('generatePrompt').addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') handleGenerate();
    });

    // Index repository
    document.getElementById('indexBtn').addEventListener('click', handleIndex);

    // Max Tokens and Temperature dropdowns (no event listeners needed - values read directly)
}

// ============================================================================
// API Health Check
// ============================================================================
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        updateStatus(data.status === 'healthy', 
            data.ollama_available ? 'Connected' : 'Ollama Unavailable');
        
        // Update available models if needed
        if (data.available_models) {
            console.log('Available models:', data.available_models);
        }
    } catch (error) {
        updateStatus(false, 'Disconnected');
        console.error('Health check failed:', error);
    }
}

function updateStatus(connected, text) {
    const wasConnected = state.connected;
    state.connected = connected;
    const indicator = document.getElementById('statusIndicator');
    const statusText = indicator.querySelector('.status-text');

    indicator.className = 'status-indicator';
    if (connected) {
        indicator.classList.add('connected');
    } else {
        indicator.classList.add('error');
    }

    statusText.textContent = text;

    // Reload models when connection status changes
    if (wasConnected !== connected) {
        console.log(`Connection status changed: ${wasConnected} -> ${connected}`);
        if (connected) {
            // Reconnected - reload models and dropdown
            loadDownloadedModels();
            // Only reload models table if on models tab
            const modelsTab = document.querySelector('[data-tab="models"]');
            if (modelsTab && modelsTab.classList.contains('active')) {
                loadModels();
            }
        }
    }
}

// ============================================================================
// Generate Code
// ============================================================================
async function handleGenerate() {
    const prompt = document.getElementById('generatePrompt').value.trim();
    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }

    const model = document.getElementById('generateModel').value;
    const maxTokens = parseInt(document.getElementById('generateMaxTokens').value);
    const temperature = parseFloat(document.getElementById('generateTemperature').value);
    const useStreaming = document.getElementById('generateStream').checked;

    const btn = document.getElementById('generateBtn');
    btn.disabled = true;

    // Show progress indicator
    showGenerateProgress(model);

    // Hide previous results
    document.getElementById('generateResult').style.display = 'none';

    try {
        // Check if streaming is enabled
        if (useStreaming) {
            await generateWithStreaming(prompt, model, maxTokens, temperature);
        } else {
            await generateWithHTTP(prompt, model, maxTokens, temperature);
        }
    } catch (error) {
        console.error('Generation error:', error);
        alert(`Error: ${error.message}`);
    } finally {
        btn.disabled = false;
        hideGenerateProgress();
    }
}

function showGenerateProgress(model) {
    // Find model display name from cache
    const modelObj = downloadedModelsCache.find(m => m.key === model);
    const displayName = modelObj ? modelObj.description : model;

    document.getElementById('generateProgressModel').textContent = displayName;
    document.getElementById('generateProgress').style.display = 'block';
}

function hideGenerateProgress() {
    document.getElementById('generateProgress').style.display = 'none';
}

async function generateWithHTTP(prompt, model, maxTokens, temperature) {
    const response = await fetch(`${API_BASE_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            prompt,
            model,
            max_tokens: maxTokens,
            temperature
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Generation failed');
    }

    const data = await response.json();
    displayGenerateResult(data);
}

async function generateWithStreaming(prompt, model, maxTokens, temperature) {
    // Prepare the result display for streaming
    const resultDiv = document.getElementById('generateResult');
    const codeElement = document.getElementById('generateCode');
    const markdownDiv = document.getElementById('generateMarkdown');
    const codePre = document.getElementById('generateCodePre');
    const warningsDiv = document.getElementById('generateWarnings');

    // Show result div immediately with markdown view
    resultDiv.style.display = 'block';
    markdownDiv.style.display = 'block';
    codePre.style.display = 'none';
    markdownDiv.innerHTML = '<p><em>Streaming...</em></p>';
    warningsDiv.style.display = 'none';
    state.viewMode = 'markdown';

    // Update progress to show streaming
    document.getElementById('generateProgressModel').textContent += ' (Streaming...)';

    let fullResponse = '';
    let startTime = Date.now();
    let lastUpdateTime = Date.now();
    const UPDATE_INTERVAL_MS = 100; // Update UI every 100ms

    try {
        // Create EventSource for SSE
        const url = new URL(`${API_BASE_URL}/generate/stream`);

        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                model,
                max_tokens: maxTokens,
                temperature
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Streaming failed');
        }

        // Read the stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            // Decode the chunk
            const chunk = decoder.decode(value, { stream: true });

            // Split by newlines to handle multiple SSE messages
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const jsonStr = line.substring(6); // Remove 'data: ' prefix

                    try {
                        const data = JSON.parse(jsonStr);

                        // Handle different event types
                        if (data.error) {
                            throw new Error(data.error);
                        }

                        if (data.status === 'started') {
                            console.log('Streaming started for model:', data.model);
                        }

                        if (data.token) {
                            // Append token to buffer
                            fullResponse += data.token;

                            // Throttle UI updates to every 100ms for better performance
                            const now = Date.now();
                            if (now - lastUpdateTime >= UPDATE_INTERVAL_MS) {
                                // Render markdown in real-time
                                renderMarkdown(fullResponse);
                                lastUpdateTime = now;

                                // Auto-scroll to bottom
                                resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                            }
                        }

                        if (data.done) {
                            // Streaming complete
                            const elapsed = (Date.now() - startTime) / 1000;

                            // Store response in state
                            state.currentRawResponse = data.response || fullResponse;

                            // Final markdown render (this also extracts code)
                            renderMarkdown(state.currentRawResponse);

                            // Update metadata
                            document.getElementById('generateTime').textContent =
                                data.total_time ? data.total_time.toFixed(2) : elapsed.toFixed(2);
                            document.getElementById('generateLength').textContent =
                                state.currentRawResponse.length;

                            console.log('Streaming complete in', elapsed, 'seconds');
                        }
                    } catch (e) {
                        console.error('Error parsing SSE data:', e, jsonStr);
                    }
                }
            }
        }

    } catch (error) {
        console.error('Streaming error:', error);
        codeElement.textContent = `// Error: ${error.message}`;
        throw error;
    }
}


function displayGenerateResult(data) {
    const resultDiv = document.getElementById('generateResult');

    // Store response in state
    state.currentRawResponse = data.response || '';

    // Render markdown (this also extracts code for code-only view)
    renderMarkdown(state.currentRawResponse);

    // Display metadata
    document.getElementById('generateTime').textContent =
        data.generation_time ? data.generation_time.toFixed(2) : '-';
    document.getElementById('generateLength').textContent = state.currentRawResponse.length;

    resultDiv.style.display = 'block';
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ============================================================================
// Index Repository
// ============================================================================
async function handleIndex() {
    const path = document.getElementById('indexPath').value.trim();
    if (!path) {
        alert('Please enter a repository path');
        return;
    }

    const parallel = document.getElementById('indexParallel').checked;
    const incremental = document.getElementById('indexIncremental').checked;
    const forceFull = document.getElementById('indexForceFull').checked;

    const btn = document.getElementById('indexBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = btn.querySelector('.btn-loader');

    btn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';

    // Show progress bar
    const progressContainer = document.getElementById('indexProgress');
    const progressBar = document.getElementById('indexProgressBar');
    const progressText = document.getElementById('indexProgressText');
    progressContainer.style.display = 'block';
    progressBar.style.width = '0%';

    // Update progress text based on mode
    const mode = forceFull ? 'FULL RE-INDEX' : (incremental ? 'INCREMENTAL' : 'STANDARD');
    progressText.textContent = `Starting ${mode} indexing...`;

    try {
        const response = await fetch(`${API_BASE_URL}/index`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                repo_path: path,
                parallel,
                incremental,
                force_full: forceFull
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Indexing failed');
        }

        const data = await response.json();
        const repoName = data.repo_path;

        // Start streaming progress updates
        streamIndexingProgress(repoName);
    } catch (error) {
        console.error('Indexing error:', error);
        alert(`Error: ${error.message}`);

        // Hide progress and reset button
        progressContainer.style.display = 'none';
        btn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

// Stream indexing progress using Server-Sent Events
function streamIndexingProgress(repoName) {
    const progressBar = document.getElementById('indexProgressBar');
    const progressText = document.getElementById('indexProgressText');
    const resultContainer = document.getElementById('indexResult');
    const resultMessage = document.getElementById('indexResultMessage');
    const btn = document.getElementById('indexBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = btn.querySelector('.btn-loader');

    const eventSource = new EventSource(`${API_BASE_URL}/index/progress/${encodeURIComponent(repoName)}/stream`);

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        // Update progress bar
        const progress = data.progress || 0;
        progressBar.style.width = `${progress}%`;
        progressText.textContent = data.message || 'Processing...';

        // Check if complete or error
        if (data.status === 'complete') {
            eventSource.close();

            // Show success message
            resultContainer.style.display = 'block';
            resultMessage.textContent = data.message || 'Repository indexed successfully!';

            // Hide progress bar after a delay
            setTimeout(() => {
                document.getElementById('indexProgress').style.display = 'none';
            }, 2000);

            // Re-enable button
            btn.disabled = false;
            btnText.style.display = 'inline';
            btnLoader.style.display = 'none';

            // Reload repositories and stats
            loadRepositories();
            loadStats();

        } else if (data.status === 'error') {
            eventSource.close();

            alert(`Indexing failed: ${data.message}`);
            document.getElementById('indexProgress').style.display = 'none';

            // Re-enable button
            btn.disabled = false;
            btnText.style.display = 'inline';
            btnLoader.style.display = 'none';
        }
    };

    eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        eventSource.close();

        // Re-enable button
        btn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    };
}

// ============================================================================
// Load Repositories
// ============================================================================
async function loadRepositories() {
    try {
        const response = await fetch(`${API_BASE_URL}/repositories`);
        if (!response.ok) return;

        const data = await response.json();
        const reposList = document.getElementById('reposList');

        if (data.success && data.repositories && data.repositories.length > 0) {
            reposList.innerHTML = data.repositories.map(repo => `
                <div class="repo-card">
                    <div class="repo-info">
                        <div class="repo-name">📁 ${repo.name}</div>
                        <div class="repo-stats">
                            <span class="repo-stat">
                                <span>📊</span>
                                <span>${repo.symbol_count.toLocaleString()} symbols</span>
                            </span>
                            <span class="repo-stat">
                                <span>📄</span>
                                <span>${repo.file_count.toLocaleString()} files</span>
                            </span>
                            <span class="repo-stat">
                                <span>🕒</span>
                                <span>${formatDate(repo.last_indexed)}</span>
                            </span>
                        </div>
                    </div>
                    <div class="repo-actions">
                        <button class="btn-small btn-secondary" onclick="viewSymbols('${repo.name}')">
                            👁️ View Symbols
                        </button>
                        <button class="btn-small btn-delete" onclick="deleteRepository('${repo.name}')">
                            🗑️ Delete
                        </button>
                    </div>
                </div>
            `).join('');
        } else {
            reposList.innerHTML = '<p class="help-text">No repositories indexed yet</p>';
        }
    } catch (error) {
        console.error('Failed to load repositories:', error);
    }
}

// ============================================================================
// View Symbols
// ============================================================================
async function viewSymbols(repoName) {
    try {
        const response = await fetch(`${API_BASE_URL}/symbols/repo/${encodeURIComponent(repoName)}?limit=100`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to load symbols');
        }

        const data = await response.json();

        if (data.success && data.symbols) {
            showSymbolsModal(repoName, data.symbols, data.total_count);
        } else {
            alert('No symbols found for this repository');
        }
    } catch (error) {
        console.error('Error loading symbols:', error);
        alert(`Error: ${error.message}`);
    }
}

function showSymbolsModal(repoName, symbols, totalCount) {
    // Create modal HTML
    const modalHTML = `
        <div id="symbolsModal" class="modal" onclick="closeSymbolsModal(event)">
            <div class="modal-content" onclick="event.stopPropagation()">
                <div class="modal-header">
                    <h2>📊 Symbols in ${repoName}</h2>
                    <button class="modal-close" onclick="closeSymbolsModal()">&times;</button>
                </div>
                <div class="modal-body">
                    <p class="help-text">Showing ${symbols.length} of ${totalCount} symbols</p>
                    <div class="symbols-table-container">
                        <table class="symbols-table">
                            <thead>
                                <tr>
                                    <th>Type</th>
                                    <th>Name</th>
                                    <th>File</th>
                                    <th>Line</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${symbols.map(symbol => `
                                    <tr>
                                        <td><span class="symbol-type">${symbol.type}</span></td>
                                        <td><code>${symbol.name}</code></td>
                                        <td class="symbol-file">${symbol.file_path}</td>
                                        <td>${symbol.line_number}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal if any
    const existingModal = document.getElementById('symbolsModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add modal to body
    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

function closeSymbolsModal(event) {
    // Close if clicking outside modal content or on close button
    if (!event || event.target.id === 'symbolsModal' || event.target.className === 'modal-close') {
        const modal = document.getElementById('symbolsModal');
        if (modal) {
            modal.remove();
        }
    }
}

// ============================================================================
// Delete Repository
// ============================================================================
async function deleteRepository(repoName) {
    if (!confirm(`Are you sure you want to delete repository "${repoName}"?\n\nThis will remove all indexed symbols and cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/repositories/${encodeURIComponent(repoName)}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to delete repository');
        }

        const data = await response.json();

        if (data.success) {
            // Reload repositories and stats
            loadRepositories();
            loadStats();

            alert(`Repository "${repoName}" deleted successfully!\n${data.symbols_deleted} symbols removed.`);
        }
    } catch (error) {
        console.error('Delete error:', error);
        alert(`Error: ${error.message}`);
    }
}

// ============================================================================
// Format Date
// ============================================================================
function formatDate(dateString) {
    if (!dateString) return 'Never';

    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} min${diffMins > 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;

    return date.toLocaleDateString();
}

// ============================================================================
// Load Statistics
// ============================================================================
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/symbols/stats`);
        if (!response.ok) return;

        const data = await response.json();

        if (data.success && data.stats) {
            document.getElementById('statSymbols').textContent = data.stats.total_symbols?.toLocaleString() || '0';
            document.getElementById('statRepos').textContent = Object.keys(data.stats.by_repo || {}).length || '0';
            document.getElementById('statFiles').textContent = data.stats.total_files || '0';
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// Model Management
// ============================================================================
// ============================================================================
// MODELS MANAGEMENT
// ============================================================================

let modelsData = [];
let downloadedModelsCache = [];

// Load downloaded models for dropdown
async function loadDownloadedModels(noCache = false) {
    try {
        // Request metadata to get display names from backend
        const params = new URLSearchParams({ metadata: 'true' });
        if (noCache) params.append('no_cache', 'true');

        const response = await fetch(`/models/downloaded?${params}`);
        const data = await response.json();
        downloadedModelsCache = data.models;

        // Update model dropdown
        updateModelDropdown();
    } catch (error) {
        console.error('Error loading downloaded models:', error);
    }
}

function updateModelDropdown() {
    const select = document.getElementById('generateModel');
    select.innerHTML = '';

    if (downloadedModelsCache.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No models downloaded - Go to Manage Models tab';
        option.disabled = true;
        select.appendChild(option);
        return;
    }

    // Use metadata from backend (no hardcoded names!)
    downloadedModelsCache.forEach((model, index) => {
        const option = document.createElement('option');
        // model is now an object with {key, name, description}
        option.value = model.key;
        option.textContent = model.description; // Use description from backend
        if (index === 0) {
            option.selected = true;  // Make first model default
        }
        select.appendChild(option);
    });
}

async function loadModels() {
    const loading = document.getElementById('modelsLoading');
    const errorDiv = document.getElementById('modelsError');

    loading.style.display = 'block';
    errorDiv.style.display = 'none';

    try {
        const response = await fetch('/models');
        const data = await response.json();

        modelsData = data.models;
        renderModelsTable(modelsData);

    } catch (error) {
        console.error('Error loading models:', error);
        errorDiv.textContent = `Error loading models: ${error.message}`;
        errorDiv.style.display = 'block';
    } finally {
        loading.style.display = 'none';
    }
}

function renderModelsTable(models) {
    const tableBody = document.getElementById('modelsTableBody');
    tableBody.innerHTML = '';

    models.forEach(model => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>
                <input type="checkbox" class="model-checkbox" data-model="${model.key}" ${model.downloaded ? '' : ''}>
            </td>
            <td class="model-name">${model.name}</td>
            <td>${model.description}</td>
            <td>${model.size}</td>
            <td>${model.license}</td>
            <td>
                <span class="model-status ${model.downloaded ? 'downloaded' : 'not-downloaded'}">
                    <span class="model-status-dot"></span>
                    ${model.downloaded ? 'Downloaded' : 'Not Downloaded'}
                </span>
            </td>
            <td>
                ${model.downloaded ? `
                    <button class="btn-icon btn-danger"
                            onclick="deleteModel('${model.key}')"
                            title="Delete model">
                        🗑️
                    </button>
                ` : ''}
            </td>
        `;
        tableBody.appendChild(row);
    });
}

// Select all checkbox
document.getElementById('selectAll').addEventListener('change', (e) => {
    const checkboxes = document.querySelectorAll('.model-checkbox');
    checkboxes.forEach(cb => cb.checked = e.target.checked);
});

// Refresh models
document.getElementById('refreshModels').addEventListener('click', loadModels);

// Download selected models
document.getElementById('downloadSelected').addEventListener('click', async () => {
    const selected = getSelectedModels();

    if (selected.length === 0) {
        showModelsMessage('Please select at least one model to download', 'error');
        return;
    }

    const successDiv = document.getElementById('modelsSuccess');
    const errorDiv = document.getElementById('modelsError');
    successDiv.style.display = 'none';
    errorDiv.style.display = 'none';

    let successCount = 0;
    let errorCount = 0;

    for (const modelKey of selected) {
        try {
            const response = await fetch(`/models/${modelKey}/download`, {
                method: 'POST'
            });

            const data = await response.json();

            if (response.ok) {
                if (data.status === 'downloading' || data.status === 'already_downloaded') {
                    successCount++;
                }
            } else {
                errorCount++;
            }
        } catch (error) {
            console.error(`Error downloading ${modelKey}:`, error);
            errorCount++;
        }
    }

    if (successCount > 0) {
        showModelsMessage(`Successfully started download for ${successCount} model(s)`, 'success');
    }

    if (errorCount > 0) {
        showModelsMessage(`Failed to download ${errorCount} model(s)`, 'error');
    }

    // Reload models after a delay to show updated status
    setTimeout(() => {
        loadModels();
    }, 2000);
});

// Delete a single model
async function deleteModel(modelKey) {
    if (!confirm(`Are you sure you want to delete the model "${modelKey}"?\n\nThis will remove the model from your local storage.`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/models/${modelKey}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (response.ok) {
            showModelsMessage(`Successfully deleted model: ${modelKey}`, 'success');

            // Reload models to update the table
            setTimeout(() => {
                loadModels();
                // Also refresh the downloaded models cache for the dropdown
                loadDownloadedModels(true);
            }, 1000);
        } else {
            showModelsMessage(`Failed to delete model: ${data.detail || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        console.error(`Error deleting model ${modelKey}:`, error);
        showModelsMessage(`Error deleting model: ${error.message}`, 'error');
    }
}

function getSelectedModels() {
    const checkboxes = document.querySelectorAll('.model-checkbox:checked');
    return Array.from(checkboxes).map(cb => cb.dataset.model);
}

function showModelsMessage(message, type) {
    const successDiv = document.getElementById('modelsSuccess');
    const errorDiv = document.getElementById('modelsError');

    if (type === 'success') {
        successDiv.textContent = message;
        successDiv.style.display = 'block';
        errorDiv.style.display = 'none';
    } else {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        successDiv.style.display = 'none';
    }
}

// Load models when the models tab is opened
document.querySelector('[data-tab="models"]').addEventListener('click', () => {
    loadModels();
});

// ============================================================================
// Auto-refresh health check
// ============================================================================
// Disabled for local development - uncomment if needed for production
// setInterval(checkHealth, 30000); // Check every 30 seconds

// ============================================================================
// Load Statistics
// ============================================================================
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/symbols/stats`);
        if (!response.ok) return;

        const data = await response.json();

        if (data.success && data.stats) {
            document.getElementById('statSymbols').textContent = data.stats.total_symbols?.toLocaleString() || '0';
            document.getElementById('statRepos').textContent = Object.keys(data.stats.by_repo || {}).length || '0';
            document.getElementById('statFiles').textContent = data.stats.total_files || '0';
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(dateString) {
    if (!dateString) return 'Never';

    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} min ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;

    return date.toLocaleDateString();
}

function copyCode(elementId) {
    const code = document.getElementById(elementId).textContent;
    navigator.clipboard.writeText(code).then(() => {
        alert('Code copied to clipboard!');
    });
}

function downloadCode(elementId, filename) {
    const code = document.getElementById(elementId).textContent;
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

function toggleResponseView() {
    const markdownDiv = document.getElementById('generateMarkdown');
    const codePre = document.getElementById('generateCodePre');
    const toggleBtn = document.getElementById('toggleViewBtn');

    if (state.viewMode === 'markdown') {
        // Switch to code-only view
        state.viewMode = 'code';
        markdownDiv.style.display = 'none';
        codePre.style.display = 'block';
        toggleBtn.textContent = '📝 Show Full Response';
    } else {
        // Switch to markdown view
        state.viewMode = 'markdown';
        markdownDiv.style.display = 'block';
        codePre.style.display = 'none';
        toggleBtn.textContent = '💻 Show Code Only';
    }
}

function extractCodeFromMarkdown(markdown) {
    // Extract code blocks from markdown
    const codeBlockRegex = /```[\w]*\n([\s\S]*?)```/g;
    const matches = [];
    let match;

    while ((match = codeBlockRegex.exec(markdown)) !== null) {
        matches.push(match[1].trim());
    }

    // If we found code blocks, join them
    if (matches.length > 0) {
        return matches.join('\n\n');
    }

    // If no code blocks found, return the whole response
    return markdown;
}

function renderMarkdown(rawResponse) {
    const markdownDiv = document.getElementById('generateMarkdown');
    const codeElement = document.getElementById('generateCode');

    // Configure marked with better options
    marked.setOptions({
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                try {
                    return hljs.highlight(code, { language: lang }).value;
                } catch (err) {
                    console.error('Highlight error:', err);
                }
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: true,
        gfm: true,
        headerIds: true,
        mangle: false
    });

    // Render markdown to HTML
    const html = marked.parse(rawResponse);
    markdownDiv.innerHTML = html;

    // Extract code for code-only view
    const extractedCode = extractCodeFromMarkdown(rawResponse);
    state.currentCode = extractedCode;
    codeElement.textContent = extractedCode;
    hljs.highlightElement(codeElement);
}
