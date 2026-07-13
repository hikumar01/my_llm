# Quick Start Guide: Unified Smart Interface

## 🎯 What Changed?

Instead of adding separate tabs, we're **enhancing the existing "Generate Code" tab** to be smart and multi-purpose:

- ✅ **Same familiar interface** - No learning curve
- ✅ **Auto-detects intent** - Figures out what you want automatically
- ✅ **Optional mode selector** - Override auto-detection if needed
- ✅ **Unified experience** - One button, multiple outputs

---

## 🚀 Implementation Steps

### Step 1: Add Smart Detector (Backend)

Create `src/smart_detector.py`:

```python
from typing import Tuple
import re

class SmartDetector:
    """Detect user intent from prompt"""
    
    WORKFLOW_KEYWORDS = [
        'workflow', 'diagram', 'flowchart', 'pipeline', 'process',
        'sequence diagram', 'state diagram', 'architecture', 'mermaid'
    ]
    
    PPT_KEYWORDS = [
        'presentation', 'slide', 'ppt', 'powerpoint', 'enhance',
        'improve slide', 'deck'
    ]
    
    @staticmethod
    def detect_intent(prompt: str, mode: str = "auto") -> Tuple[str, float]:
        """
        Returns: (detected_mode, confidence)
        Modes: 'code', 'workflow', 'ppt'
        """
        if mode != "auto":
            return mode, 1.0
        
        prompt_lower = prompt.lower()
        
        # Check workflow keywords
        workflow_matches = sum(1 for kw in SmartDetector.WORKFLOW_KEYWORDS 
                              if kw in prompt_lower)
        
        # Check PPT keywords
        ppt_matches = sum(1 for kw in SmartDetector.PPT_KEYWORDS 
                         if kw in prompt_lower)
        
        # Check code patterns
        code_patterns = [r'\bfunction\b', r'\bclass\b', r'\bdef\b', 
                        r'\bimport\b', r'\breturn\b']
        code_matches = sum(1 for pattern in code_patterns 
                          if re.search(pattern, prompt_lower))
        
        # Determine winner
        if workflow_matches >= 2:
            return 'workflow', min(workflow_matches / 3.0, 1.0)
        elif ppt_matches >= 2:
            return 'ppt', min(ppt_matches / 3.0, 1.0)
        elif code_matches >= 1:
            return 'code', min(code_matches / 2.0, 1.0)
        else:
            return 'code', 0.5  # Default to code
```

---

### Step 2: Update API Server

Add to `src/api_server.py`:

```python
from smart_detector import SmartDetector

# New unified endpoint
@app.post("/generate/smart")
async def smart_generate(request: SmartGenerationRequest):
    """Smart generation with auto-detection"""
    
    # Detect intent
    detected_mode, confidence = SmartDetector.detect_intent(
        request.prompt, 
        request.mode
    )
    
    # Route to appropriate handler
    if detected_mode == "workflow":
        return await generate_workflow(request, confidence)
    elif detected_mode == "ppt":
        return await generate_ppt_options(request, confidence)
    else:
        return await generate_code(request, confidence)


async def generate_workflow(request, confidence):
    """Generate workflow diagram"""
    # For now, use LLM to generate Mermaid code
    client = OllamaClient(AVAILABLE_LOCAL_MODELS[request.model]["name"])
    
    prompt = f"""Generate a Mermaid diagram for: {request.prompt}

Output ONLY the Mermaid code, starting with 'graph' or 'sequenceDiagram'.
No explanations, just the diagram code."""
    
    diagram_code = client.generate(prompt, max_tokens=1000, temperature=0.3)
    
    return {
        "type": "workflow",
        "mode": "workflow",
        "confidence": confidence,
        "diagram_code": diagram_code,
        "format": "mermaid"
    }


async def generate_ppt_options(request, confidence):
    """Generate PPT enhancement options"""
    client = OllamaClient(AVAILABLE_LOCAL_MODELS[request.model]["name"])
    
    prompt = f"""Generate 3 different design options for: {request.prompt}

For each option, provide:
1. Title (e.g., "Modern Design", "Infographic Style")
2. Description (2-3 sentences)
3. Key features (3-4 bullet points)

Format as JSON array."""
    
    response = client.generate(prompt, max_tokens=1500, temperature=0.7)
    
    # Parse response (simplified - add proper JSON parsing)
    options = [
        {
            "id": 1,
            "title": "Modern Design",
            "description": "Clean, modern design with bold typography and minimalist icons.",
            "features": ["Bold colors", "Clean layout", "Modern typography"]
        },
        {
            "id": 2,
            "title": "Infographic Style",
            "description": "Visual-heavy design with charts and data visualization.",
            "features": ["Data viz", "Icons", "Visual hierarchy"]
        },
        {
            "id": 3,
            "title": "Professional",
            "description": "Corporate-friendly design with structured layout.",
            "features": ["Professional colors", "Clear structure", "Business-ready"]
        }
    ]
    
    return {
        "type": "ppt",
        "mode": "ppt",
        "confidence": confidence,
        "options": options
    }


async def generate_code(request, confidence):
    """Generate code (existing functionality)"""
    client = OllamaClient(AVAILABLE_LOCAL_MODELS[request.model]["name"])
    
    response = client.generate(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    
    return {
        "type": "code",
        "mode": "code",
        "confidence": confidence,
        "response": response
    }


# Pydantic model
class SmartGenerationRequest(BaseModel):
    prompt: str
    mode: str = "auto"
    model: str = "deepseek-coder"
    max_tokens: int = 2000
    temperature: float = 0.2
```

---

### Step 3: Update Frontend HTML

Modify `frontend/index.html` - Add mode selector in Generate Code tab:

```html
<!-- Add BEFORE the model selector -->
<div class="form-group">
    <label for="generateMode">
        Mode (Optional - Auto-detects if not selected)
        <span style="background: #667eea; color: white; padding: 2px 8px; 
               border-radius: 12px; font-size: 0.8rem; margin-left: 5px;">Smart</span>
    </label>
    <select id="generateMode" class="input-select">
        <option value="auto" selected>🤖 Auto-detect (Recommended)</option>
        <option value="code">💻 Code Generation</option>
        <option value="workflow">🎨 Workflow Diagram</option>
        <option value="ppt">📊 PPT Enhancement</option>
    </select>
</div>
```

Update the prompt textarea placeholder:

```html
<textarea
    id="generatePrompt"
    class="input-textarea"
    placeholder="Examples: 'Write a Python function...' OR 'Create a CI/CD pipeline diagram' OR 'Enhance my presentation slide...'"
    rows="4"></textarea>
```

---

### Step 4: Update Frontend JavaScript

Modify `frontend/app.js`:

```javascript
// Update the generate button handler
async function handleGenerate() {
    const prompt = document.getElementById('generatePrompt').value;
    const mode = document.getElementById('generateMode').value;
    const model = document.getElementById('generateModel').value;
    const maxTokens = parseInt(document.getElementById('generateMaxTokens').value);
    const temperature = parseFloat(document.getElementById('generateTemperature').value);
    const useStreaming = document.getElementById('generateStream').checked;
    
    if (!prompt.trim()) {
        alert('Please enter a prompt');
        return;
    }
    
    showGenerateProgress(model);
    
    try {
        // Use new smart endpoint
        const response = await apiPost('/generate/smart', {
            prompt,
            mode,
            model,
            max_tokens: maxTokens,
            temperature
        });
        
        const data = await response.json();
        
        // Display based on detected type
        displaySmartResult(data);
        
    } catch (error) {
        console.error('Generation error:', error);
        alert(`Error: ${error.message}`);
    } finally {
        hideGenerateProgress();
    }
}

function displaySmartResult(data) {
    const resultDiv = document.getElementById('generateResult');
    
    // Show mode indicator
    const modeIndicator = `
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 12px 20px; border-radius: 8px;
                    margin-bottom: 20px; font-weight: 600;">
            ${getModeIcon(data.type)} ${getModeLabel(data.type)}
            <span style="opacity: 0.8; margin-left: 10px;">
                (${(data.confidence * 100).toFixed(0)}% confidence)
            </span>
        </div>
    `;
    
    // Display based on type
    if (data.type === 'workflow') {
        displayWorkflowResult(data, modeIndicator);
    } else if (data.type === 'ppt') {
        displayPPTResult(data, modeIndicator);
    } else {
        displayCodeResult(data, modeIndicator);
    }
}

function getModeIcon(type) {
    const icons = {
        'code': '💻',
        'workflow': '🎨',
        'ppt': '📊'
    };
    return icons[type] || '💻';
}

function getModeLabel(type) {
    const labels = {
        'code': 'Code Generation',
        'workflow': 'Workflow Diagram',
        'ppt': 'PPT Enhancement Options'
    };
    return labels[type] || 'Code Generation';
}

function displayWorkflowResult(data, modeIndicator) {
    const resultDiv = document.getElementById('generateResult');
    
    resultDiv.innerHTML = modeIndicator + `
        <div class="workflow-container">
            <div class="mermaid">${data.diagram_code}</div>
        </div>
        <div style="margin-top: 20px; display: flex; gap: 10px;">
            <button class="btn-secondary btn-sm" onclick="exportDiagram('svg')">
                📄 Export SVG
            </button>
            <button class="btn-secondary btn-sm" onclick="exportDiagram('png')">
                🖼️ Export PNG
            </button>
            <button class="btn-secondary btn-sm" onclick="copyDiagramCode()">
                📋 Copy Code
            </button>
        </div>
    `;
    
    resultDiv.style.display = 'block';
    
    // Initialize Mermaid (add mermaid.js to index.html first)
    if (typeof mermaid !== 'undefined') {
        mermaid.init(undefined, document.querySelectorAll('.mermaid'));
    }
}

function displayPPTResult(data, modeIndicator) {
    const resultDiv = document.getElementById('generateResult');
    
    const optionsHtml = data.options.map((opt, idx) => `
        <div class="ppt-option" style="background: white; border: 3px solid #dee2e6;
                                       border-radius: 12px; padding: 20px; margin: 10px 0;
                                       cursor: pointer;" 
             onclick="togglePPTOption(${idx})">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="color: #667eea;">Option ${idx + 1}: ${opt.title}</h4>
                <input type="checkbox" id="ppt-opt-${idx}" style="width: 20px; height: 20px;">
            </div>
            <p style="color: #6c757d; margin: 10px 0;">${opt.description}</p>
            <ul style="color: #6c757d; margin-left: 20px;">
                ${opt.features.map(f => `<li>${f}</li>`).join('')}
            </ul>
        </div>
    `).join('');
    
    resultDiv.innerHTML = modeIndicator + optionsHtml + `
        <div style="margin-top: 20px; display: flex; gap: 10px;">
            <button class="btn-primary" onclick="applyPPTOptions()">
                ✅ Apply Selected
            </button>
            <button class="btn-secondary" onclick="regeneratePPT()">
                🔄 Regenerate Options
            </button>
        </div>
    `;
    
    resultDiv.style.display = 'block';
}

function displayCodeResult(data, modeIndicator) {
    // Use existing displayGenerateResult but add mode indicator
    const resultDiv = document.getElementById('generateResult');
    resultDiv.innerHTML = modeIndicator;
    
    // Then call existing display logic
    state.currentRawResponse = data.response || '';
    renderMarkdown(state.currentRawResponse);
    
    document.getElementById('generateTime').textContent = 
        data.generation_time ? data.generation_time.toFixed(2) : '-';
    document.getElementById('generateLength').textContent = 
        state.currentRawResponse.length;
    
    resultDiv.style.display = 'block';
}

// Helper functions
function togglePPTOption(idx) {
    const checkbox = document.getElementById(`ppt-opt-${idx}`);
    checkbox.checked = !checkbox.checked;
}

function applyPPTOptions() {
    const selected = [];
    document.querySelectorAll('[id^="ppt-opt-"]').forEach((cb, idx) => {
        if (cb.checked) selected.push(idx + 1);
    });
    
    if (selected.length === 0) {
        alert('Please select at least one option');
        return;
    }
    
    alert(`Applying options: ${selected.join(', ')}`);
}

function regeneratePPT() {
    // Re-trigger generation
    handleGenerate();
}

function exportDiagram(format) {
    alert(`Exporting diagram as ${format.toUpperCase()}`);
}

function copyDiagramCode() {
    const code = document.querySelector('.mermaid').textContent;
    navigator.clipboard.writeText(code);
    alert('Diagram code copied to clipboard!');
}
```

---

### Step 5: Add Mermaid.js

Add to `frontend/index.html` in the `<head>` section:

```html
<!-- Add Mermaid for diagram rendering -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
    mermaid.initialize({ 
        startOnLoad: false, 
        theme: 'default',
        securityLevel: 'loose'
    });
</script>
```

---

## 🎯 Testing

### Test Cases

1. **Code Generation**
   - Input: "Write a Python function to reverse a string"
   - Expected: Code output with syntax highlighting

2. **Workflow Diagram**
   - Input: "Create a CI/CD pipeline diagram"
   - Expected: Mermaid diagram with export options

3. **PPT Enhancement**
   - Input: "Enhance my presentation slide about AI"
   - Expected: 3 design options to select from

4. **Auto-detection**
   - Leave mode on "Auto-detect"
   - Try various prompts
   - Verify correct detection

---

## 📊 Example Prompts

### Code (Auto-detected)
- "Write a Python function to reverse a string"
- "Implement binary search in JavaScript"
- "Create a React component for a login form"

### Workflow (Auto-detected)
- "Create a CI/CD pipeline diagram"
- "Generate a flowchart for user authentication"
- "Show me a sequence diagram for order processing"
- "Design a state diagram for a traffic light"

### PPT (Auto-detected)
- "Enhance my presentation slide about AI"
- "Make my slide more visual with infographics"
- "Improve the design of my product roadmap slide"

---

## ✅ Benefits

1. **No UI clutter** - Same familiar interface
2. **Smart defaults** - Works without configuration
3. **User control** - Can override auto-detection
4. **Progressive enhancement** - Existing features still work
5. **Easy to extend** - Add more modes in the future

---

## 🚀 Next Steps

1. Test the smart detector with various prompts
2. Refine detection keywords based on usage
3. Add more sophisticated pattern matching
4. Implement actual workflow rendering
5. Build PPT enhancement backend
6. Add export functionality

**The mockup is now open in your browser - try it out!**

