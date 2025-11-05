// ============================================================================
// Configuration
// ============================================================================
const API_BASE_URL = window.location.origin;

// ============================================================================
// State Management
// ============================================================================
const state = {
    currentTab: 'assistant',
    theme: localStorage.getItem('theme') || 'light',
    connected: false,
    ws: null,
    currentTaskType: 'code', // 'code', 'text', or 'ppt'
    currentRawResponse: '',
    selectedModel: null
};

// ============================================================================
// API Helper Functions
// ============================================================================
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const response = await fetch(url, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options
    });

    if (!response.ok && options.throwOnError !== false) {
        const error = await response.json().catch(() => ({ detail: 'Request failed' }));
        throw new Error(error.detail || `Request failed: ${response.status}`);
    }

    return response;
}

async function apiGet(endpoint) {
    const response = await apiRequest(endpoint, { throwOnError: false });
    return response.ok ? await response.json() : null;
}

async function apiPost(endpoint, data) {
    return await apiRequest(endpoint, {
        method: 'POST',
        body: JSON.stringify(data)
    });
}

async function apiDelete(endpoint) {
    return await apiRequest(endpoint, { method: 'DELETE' });
}

// ============================================================================
// Initialization
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initTabs();
    initEventListeners();
    initMermaid();
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
    updateMermaidTheme();
}

function updateThemeIcon() {
    const icon = document.querySelector('.theme-icon');
    icon.textContent = state.theme === 'light' ? '🌙' : '☀️';
}

// ============================================================================
// Mermaid Initialization
// ============================================================================
function initMermaid() {
    if (typeof mermaid !== 'undefined') {
        mermaid.initialize({
            startOnLoad: false,
            theme: state.theme === 'dark' ? 'dark' : 'default',
            securityLevel: 'loose',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }
        });
        console.log('Mermaid initialized');
    }
}

// Update Mermaid theme when app theme changes
function updateMermaidTheme() {
    if (typeof mermaid !== 'undefined') {
        mermaid.initialize({
            theme: state.theme === 'dark' ? 'dark' : 'default'
        });
        // Re-render any existing diagrams
        renderMermaidDiagrams();
    }
}

// Render all Mermaid diagrams in the document
async function renderMermaidDiagrams() {
    if (typeof mermaid === 'undefined') return;

    // Find all code blocks with language 'mermaid'
    const mermaidBlocks = document.querySelectorAll('pre code.language-mermaid');

    for (let i = 0; i < mermaidBlocks.length; i++) {
        const block = mermaidBlocks[i];
        const pre = block.parentElement;

        // Skip if already rendered
        if (pre.classList.contains('mermaid-rendered')) continue;

        const code = block.textContent;

        try {
            // Create a div to hold the rendered diagram
            const div = document.createElement('div');
            div.className = 'mermaid';
            div.textContent = code;

            // Replace the pre block with the mermaid div
            pre.replaceWith(div);

            // Render the diagram
            await mermaid.run({
                nodes: [div]
            });

        } catch (error) {
            console.error('Mermaid rendering error:', error);
            // Keep the original code block if rendering fails
        }
    }
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

    // Load models when switching to models tab
    if (tabName === 'models') {
        loadModels();
    }
}

// ============================================================================
// Event Listeners
// ============================================================================
function initEventListeners() {
    // Theme toggle
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);

    // Header title - click to go to AI Assistant
    document.getElementById('headerTitle').addEventListener('click', () => {
        switchTab('assistant');
    });

    // Status indicator - click to open models tab
    document.getElementById('statusIndicator').addEventListener('click', () => {
        switchTab('models');
    });

    // Index repository icon - click to open index tab
    document.getElementById('indexRepoBtn').addEventListener('click', () => {
        switchTab('index');
    });

    // Task type selector buttons
    document.querySelectorAll('.task-type-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const taskType = btn.dataset.task;
            selectTaskType(taskType);
        });
    });

    // Assistant submit button
    document.getElementById('assistantSubmitBtn').addEventListener('click', handleAssistantSubmit);

    // Prompt keyboard shortcut
    document.getElementById('assistantPrompt').addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') handleAssistantSubmit();
    });

    // Index repository
    document.getElementById('indexBtn').addEventListener('click', handleIndex);
}

// ============================================================================
// API Health Check
// ============================================================================
async function checkHealth() {
    try {
        const data = await apiGet('/health');
        if (data) {
            updateStatus(data.status === 'healthy',
                data.ollama_available ? 'Connected' : 'Ollama Unavailable');

            // Update available models if needed
            if (data.available_models) {
                console.log('Available models:', data.available_models);
            }
        } else {
            updateStatus(false, 'Disconnected');
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
    const statusIcon = indicator.querySelector('.status-icon');

    indicator.className = 'status-indicator';

    // Update icon, class, and tooltip based on status
    if (text === 'Connecting...') {
        indicator.classList.add('connecting');
        statusIcon.textContent = '🔄';  // Spinning arrows for connecting
        indicator.title = 'Connecting to Ollama... (Click to manage models)';
    } else if (connected) {
        indicator.classList.add('connected');
        statusIcon.textContent = '✅';  // Green checkmark for connected
        indicator.title = 'Connected to Ollama (Click to manage models)';
    } else {
        indicator.classList.add('error');
        statusIcon.textContent = '❌';  // Red X for disconnected
        indicator.title = 'Disconnected - Ollama not available (Click to manage models)';
    }

    // Hide text, show only icon
    statusText.style.display = 'none';

    // Reload models when connection status changes
    if (wasConnected !== connected) {
        console.log(`Connection status changed: ${wasConnected} -> ${connected}`);
        if (connected) {
            // Reconnected - reload models and dropdown
            loadDownloadedModels();
            // Only reload models table if on models tab
            const modelsContent = document.getElementById('models');
            if (modelsContent && modelsContent.classList.contains('active')) {
                loadModels();
            }
        }
    }
}

// ============================================================================
// Unified AI Assistant
// ============================================================================

function selectTaskType(taskType) {
    state.currentTaskType = taskType;

    // Update button states
    document.querySelectorAll('.task-type-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.task === taskType);
    });

    // Update config panel
    document.querySelectorAll('.task-config').forEach(config => {
        config.classList.remove('active');
    });
    document.getElementById(`${taskType}Config`).classList.add('active');
}

async function handleAssistantSubmit() {
    const prompt = document.getElementById('assistantPrompt').value.trim();

    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }

    const taskType = state.currentTaskType;

    // Get configuration based on task type
    let config = {};
    if (taskType === 'code') {
        config = {
            max_tokens: parseInt(document.getElementById('codeMaxTokens').value),
            temperature: parseFloat(document.getElementById('codeTemperature').value),
            stream: true  // Always stream
        };
    } else if (taskType === 'text') {
        config = {
            style: document.getElementById('textStyle').value,
            max_tokens: parseInt(document.getElementById('textMaxTokens').value),
            stream: true  // Always stream
        };
    } else if (taskType === 'ppt') {
        config = {
            output_type: document.getElementById('pptOutputType').value,
            max_tokens: parseInt(document.getElementById('pptMaxTokens').value),
            stream: true  // Always stream
        };
    }

    // Show progress
    document.getElementById('assistantProgress').style.display = 'block';
    document.getElementById('assistantResult').style.display = 'none';
    document.getElementById('assistantSubmitBtn').disabled = true;

    const startTime = Date.now();

    try {
        // Always use streaming
        await handleAssistantStream(prompt, taskType, config, startTime);
    } catch (error) {
        console.error('Generation error:', error);
        alert(`Error: ${error.message}`);
        document.getElementById('assistantProgress').style.display = 'none';
    } finally {
        document.getElementById('assistantSubmitBtn').disabled = false;
    }
}

async function handleAssistantNonStream(prompt, taskType, config, startTime) {
    const response = await apiPost('/assistant/generate', {
        prompt,
        task_type: taskType,
        ...config
    });

    const data = await response.json();

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);

    // Hide progress, show result
    document.getElementById('assistantProgress').style.display = 'none';
    document.getElementById('assistantResult').style.display = 'block';

    // Render result
    const markdown = document.getElementById('assistantMarkdown');
    markdown.innerHTML = marked.parse(data.response || data.result || '');

    // Highlight code blocks
    markdown.querySelectorAll('pre code').forEach(block => {
        hljs.highlightElement(block);
    });

    // Render Mermaid diagrams
    renderMermaidDiagrams();

    // Update metadata
    document.getElementById('assistantModel').textContent = data.model || 'Unknown';
    document.getElementById('assistantTime').textContent = elapsed;
    document.getElementById('assistantLength').textContent = (data.response || data.result || '').length;

    state.currentRawResponse = data.response || data.result || '';
}

async function handleAssistantStream(prompt, taskType, config, startTime) {
    // Use fetch directly for streaming to avoid JSON parsing issues
    const response = await fetch(`${API_BASE_URL}/assistant/generate/stream`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt,
            task_type: taskType,
            ...config
        })
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Request failed' }));
        throw new Error(error.detail || `Request failed: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullResponse = '';
    let modelName = '';

    // Show result container immediately for streaming
    document.getElementById('assistantProgress').style.display = 'none';
    document.getElementById('assistantResult').style.display = 'block';

    const markdown = document.getElementById('assistantMarkdown');
    markdown.innerHTML = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
            if (!line.trim() || !line.startsWith('data: ')) continue;

            const data = line.slice(6);
            if (data === '[DONE]') continue;

            try {
                const parsed = JSON.parse(data);
                if (parsed.token) {
                    fullResponse += parsed.token;
                    markdown.innerHTML = marked.parse(fullResponse);
                    markdown.querySelectorAll('pre code').forEach(block => {
                        hljs.highlightElement(block);
                    });
                    // Render Mermaid diagrams
                    renderMermaidDiagrams();
                }
                if (parsed.model) {
                    modelName = parsed.model;
                }
            } catch (e) {
                console.error('Parse error:', e);
            }
        }
    }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);

    // Update metadata
    document.getElementById('assistantModel').textContent = modelName || 'Unknown';
    document.getElementById('assistantTime').textContent = elapsed;
    document.getElementById('assistantLength').textContent = fullResponse.length;

    state.currentRawResponse = fullResponse;
}

function copyAssistantResult() {
    navigator.clipboard.writeText(state.currentRawResponse);
    alert('Copied to clipboard!');
}

function downloadAssistantResult() {
    const taskType = state.currentTaskType;
    const extension = taskType === 'code' ? 'txt' : 'md';
    const blob = new Blob([state.currentRawResponse], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `assistant_result.${extension}`;
    a.click();
    URL.revokeObjectURL(url);
}

// ============================================================================
// Old Generate Code functions removed - using unified AI Assistant interface
// ============================================================================



// ============================================================================
// Old Text Enhancement functions removed - using unified AI Assistant interface
// ============================================================================

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
        const response = await apiPost('/index', {
            repo_path: path,
            parallel,
            incremental,
            force_full: forceFull
        });

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
        const data = await apiGet('/repositories');
        const reposList = document.getElementById('reposList');

        if (data?.success && data.repositories?.length > 0) {
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
        const data = await apiGet(`/symbols/repo/${encodeURIComponent(repoName)}?limit=100`);

        if (data?.success && data.symbols) {
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
        const response = await apiDelete(`/repositories/${encodeURIComponent(repoName)}`);
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
        const data = await apiGet('/symbols/stats');
        if (data?.success && data.stats) {
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

function toggleDisplay(elementId, show) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.display = show ? 'block' : 'none';
    }
}

function hideElements(...elementIds) {
    elementIds.forEach(id => toggleDisplay(id, false));
}

function showElement(elementId) {
    toggleDisplay(elementId, true);
}

// ============================================================================
// Model Management
// ============================================================================
let modelsData = [];
let downloadedModelsCache = [];

// Load downloaded models for dropdown
async function loadDownloadedModels(noCache = false) {
    try {
        // Request metadata to get display names from backend
        const params = new URLSearchParams({ metadata: 'true' });
        if (noCache) params.append('no_cache', 'true');

        const data = await apiGet(`/models/downloaded?${params}`);
        if (data) {
            downloadedModelsCache = data.models;
            updateModelDropdown();
        }
    } catch (error) {
        console.error('Error loading downloaded models:', error);
    }
}

function updateModelDropdown() {
    const generateSelect = document.getElementById('generateModel');
    const enhanceSelect = document.getElementById('enhanceModel');

    // Check if elements exist (they don't in the new unified interface)
    if (!generateSelect || !enhanceSelect) {
        return;
    }

    generateSelect.innerHTML = '';
    enhanceSelect.innerHTML = '';

    if (downloadedModelsCache.length === 0) {
        const option1 = document.createElement('option');
        option1.value = '';
        option1.textContent = 'No models downloaded - Go to Manage Models tab';
        option1.disabled = true;
        generateSelect.appendChild(option1);

        const option2 = document.createElement('option');
        option2.value = '';
        option2.textContent = 'No models downloaded - Go to Manage Models tab';
        option2.disabled = true;
        enhanceSelect.appendChild(option2);
        return;
    }

    // Use metadata from backend (no hardcoded names!)
    downloadedModelsCache.forEach((model, index) => {
        // Add to generate model dropdown
        const option1 = document.createElement('option');
        option1.value = model.key;
        option1.textContent = model.description; // Use description from backend
        if (index === 0) {
            option1.selected = true;  // Make first model default
        }
        generateSelect.appendChild(option1);

        // Add to enhance model dropdown
        const option2 = document.createElement('option');
        option2.value = model.key;
        option2.textContent = model.description;
        if (index === 0) {
            option2.selected = true;
        }
        enhanceSelect.appendChild(option2);
    });
}

async function loadModels() {
    showElement('modelsLoading');
    hideElements('modelsError');

    try {
        const data = await apiGet('/models');
        if (data) {
            modelsData = data.models;
            renderModelsTable(modelsData);
        }
    } catch (error) {
        console.error('Error loading models:', error);
        const errorDiv = document.getElementById('modelsError');
        errorDiv.textContent = `Error loading models: ${error.message}`;
        showElement('modelsError');
    } finally {
        hideElements('modelsLoading');
    }
}

function renderModelsTable(models) {
    const tableBody = document.getElementById('modelsTableBody');
    tableBody.innerHTML = '';

    // Sort models: recommended first, then by priority
    const sortedModels = [...models].sort((a, b) => {
        if (a.recommended && !b.recommended) return -1;
        if (!a.recommended && b.recommended) return 1;
        return (a.priority || 999) - (b.priority || 999);
    });

    sortedModels.forEach(model => {
        const row = document.createElement('tr');

        // Highlight recommended models
        if (model.recommended) {
            row.style.background = 'linear-gradient(90deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%)';
            row.style.borderLeft = '3px solid #667eea';
        }

        // Get best-for badge
        const bestForBadges = {
            'code': '💻 Code',
            'workflow': '🎨 Workflow',
            'ppt': '📊 PPT'
        };
        const bestForBadge = bestForBadges[model.best_for] || '🔧 General';

        // Get performance summary
        const perfSummary = model.performance
            ? `Code: ${model.performance.code}<br>Workflow: ${model.performance.workflow}<br>PPT: ${model.performance.ppt}<br>Speed: ${model.performance.speed}`
            : 'N/A';

        row.innerHTML = `
            <td>
                <input type="checkbox" class="model-checkbox" data-model="${model.key}" ${model.downloaded ? '' : ''}>
            </td>
            <td class="model-name">
                ${model.name}
                ${model.recommended ? '<span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; margin-left: 8px;">⭐ Recommended</span>' : ''}
            </td>
            <td>${model.description}</td>
            <td>
                <span style="background: #667eea; color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; font-weight: 600;">
                    ${bestForBadge}
                </span>
            </td>
            <td>${model.size}</td>
            <td style="font-size: 0.85rem; line-height: 1.6;">${perfSummary}</td>
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

// Download recommended models
document.getElementById('downloadRecommended').addEventListener('click', async () => {
    // Get recommended models that are not downloaded
    const recommendedModels = modelsData
        .filter(m => m.recommended && !m.downloaded)
        .map(m => m.key);

    if (recommendedModels.length === 0) {
        showModelsMessage('All recommended models are already downloaded!', 'success');
        return;
    }

    const recommendedInfo = modelsData
        .filter(m => m.recommended && !m.downloaded)
        .map(m => `• ${m.name} (${m.size}) - Best for ${m.best_for}`)
        .join('\n');

    const confirmMsg = `This will download ${recommendedModels.length} recommended model(s):\n\n${recommendedInfo}\n\nTotal size: ~10.5 GB\nContinue?`;

    if (!confirm(confirmMsg)) {
        return;
    }

    hideElements('modelsSuccess', 'modelsError');

    let successCount = 0;
    let errorCount = 0;

    for (const modelKey of recommendedModels) {
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
        showModelsMessage(
            `Started downloading ${successCount} recommended model(s). Check logs for progress. This may take 10-20 minutes.`,
            'success'
        );
    }

    if (errorCount > 0) {
        showModelsMessage(`Failed to download ${errorCount} model(s)`, 'error');
    }

    // Reload models after a delay
    setTimeout(() => {
        loadModels();
    }, 2000);
});

// Download selected models
document.getElementById('downloadSelected').addEventListener('click', async () => {
    const selected = getSelectedModels();

    if (selected.length === 0) {
        showModelsMessage('Please select at least one model to download', 'error');
        return;
    }

    hideElements('modelsSuccess', 'modelsError');

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
        await apiDelete(`/models/${modelKey}`);
        showModelsMessage(`Successfully deleted model: ${modelKey}`, 'success');

        // Reload models to update the table
        setTimeout(() => {
            loadModels();
            // Also refresh the downloaded models cache for the dropdown
            loadDownloadedModels(true);
        }, 1000);
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

// Load models when the models tab is opened (via status icon click)
// Note: Models are loaded when switching to the models tab via switchTab() function

// ============================================================================
// Auto-refresh health check
// ============================================================================
// Disabled for local development - uncomment if needed for production
// setInterval(checkHealth, 30000); // Check every 30 seconds



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

// Old view toggle and markdown rendering functions removed - no longer needed
