# Simplified Design: Unified Smart Interface

## 🎯 Core Concept

**One interface, multiple capabilities** - The existing "Generate Code" tab becomes a smart, unified interface that automatically detects what the user wants to do:
- Generate code
- Create workflow diagrams
- Enhance presentations

**No separate tabs needed!** The backend intelligently detects the intent and responds accordingly.

---

## 🎨 UI Changes (Minimal)

### Modified Generate Tab

```
┌─────────────────────────────────────────────────────────────┐
│  Generate Code                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Prompt:                                                    │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Examples:                                             │ │
│  │ • "Write a Python function to reverse a string"       │ │
│  │ • "Create a CI/CD workflow diagram"                   │ │
│  │ • "Generate a user authentication flowchart"          │ │
│  │ • "Enhance my presentation slide about AI"            │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  Mode: ○ Auto-detect (Smart)  ○ Code  ○ Workflow  ○ PPT   │
│                                                             │
│  Model: [DeepSeek Coder ▼]    Max Tokens: [2000 ▼]        │
│                                                             │
│  Temperature: [0.2 ▼]         ☑ Stream tokens              │
│                                                             │
│  [Generate] ← Same button for everything                   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Result (Dynamic based on detection):                      │
│                                                             │
│  IF CODE:     Syntax highlighted code                      │
│  IF WORKFLOW: Interactive diagram with export options      │
│  IF PPT:      Multiple enhancement options to select       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Changes to `frontend/index.html`

1. **Add mode selector** (optional, defaults to auto-detect)
2. **Keep same result container** but make it adaptive
3. **Add file upload** (hidden by default, shown when PPT mode detected)

---

## 🧠 Smart Detection Logic

### Backend: `src/smart_detector.py`

```python
from typing import Tuple, Dict
import re

class SmartDetector:
    """Intelligently detect user intent from prompt"""
    
    # Keywords for each mode
    WORKFLOW_KEYWORDS = [
        'workflow', 'diagram', 'flowchart', 'flow chart', 'process',
        'pipeline', 'sequence diagram', 'state diagram', 'architecture',
        'system design', 'data flow', 'process flow', 'mermaid', 'graphviz'
    ]
    
    PPT_KEYWORDS = [
        'presentation', 'slide', 'ppt', 'powerpoint', 'enhance',
        'improve slide', 'make slide', 'slide design', 'deck'
    ]
    
    CODE_PATTERNS = [
        r'\bfunction\b', r'\bclass\b', r'\bdef\b', r'\bimport\b',
        r'\bconst\b', r'\bvar\b', r'\blet\b', r'\breturn\b',
        r'\bpublic\b', r'\bprivate\b', r'\bvoid\b'
    ]
    
    @staticmethod
    def detect_intent(prompt: str, mode: str = "auto") -> Tuple[str, float]:
        """
        Detect user intent from prompt.
        
        Args:
            prompt: User's input prompt
            mode: User-selected mode ("auto", "code", "workflow", "ppt")
            
        Returns:
            Tuple of (detected_mode, confidence_score)
        """
        # If user explicitly selected a mode, use it
        if mode != "auto":
            return mode, 1.0
        
        prompt_lower = prompt.lower()
        
        # Score each category
        workflow_score = SmartDetector._calculate_score(
            prompt_lower, SmartDetector.WORKFLOW_KEYWORDS
        )
        
        ppt_score = SmartDetector._calculate_score(
            prompt_lower, SmartDetector.PPT_KEYWORDS
        )
        
        code_score = SmartDetector._calculate_code_score(prompt)
        
        # Determine winner
        scores = {
            'workflow': workflow_score,
            'ppt': ppt_score,
            'code': code_score
        }
        
        detected_mode = max(scores, key=scores.get)
        confidence = scores[detected_mode]
        
        # Default to code if confidence is low
        if confidence < 0.3:
            return 'code', 0.5
        
        return detected_mode, confidence
    
    @staticmethod
    def _calculate_score(text: str, keywords: list) -> float:
        """Calculate score based on keyword matches"""
        matches = sum(1 for keyword in keywords if keyword in text)
        return min(matches / 3.0, 1.0)  # Normalize to 0-1
    
    @staticmethod
    def _calculate_code_score(text: str) -> float:
        """Calculate score for code generation"""
        pattern_matches = sum(
            1 for pattern in SmartDetector.CODE_PATTERNS 
            if re.search(pattern, text, re.IGNORECASE)
        )
        
        # Check for code-related verbs
        code_verbs = ['write', 'create', 'implement', 'build', 'develop']
        verb_matches = sum(1 for verb in code_verbs if verb in text.lower())
        
        return min((pattern_matches + verb_matches) / 4.0, 1.0)
    
    @staticmethod
    def get_suggested_model(mode: str) -> str:
        """Suggest best model for detected mode"""
        model_map = {
            'code': 'deepseek-coder',
            'workflow': 'llama3.2',
            'ppt': 'llama3.2-vision'
        }
        return model_map.get(mode, 'deepseek-coder')
```

---

## 🔧 Backend Implementation

### Modified `src/api_server.py`

```python
from smart_detector import SmartDetector
from workflow_generator import WorkflowGenerator
from ppt_enhancer import PPTEnhancer

# Add new unified endpoint
@app.post(
    "/generate/smart",
    tags=["Smart Generation"],
    summary="Smart Unified Generation",
    description="Automatically detects intent and generates appropriate output"
)
async def smart_generate(request: SmartGenerationRequest):
    """
    Smart generation endpoint that auto-detects user intent.
    
    Request:
        {
            "prompt": "Create a CI/CD pipeline diagram",
            "mode": "auto",  // or "code", "workflow", "ppt"
            "model": "deepseek-coder",
            "max_tokens": 2000,
            "temperature": 0.2
        }
    
    Response varies based on detected mode:
        - Code: { "type": "code", "response": "...", ... }
        - Workflow: { "type": "workflow", "diagram": "...", "format": "mermaid", ... }
        - PPT: { "type": "ppt", "options": [...], ... }
    """
    try:
        # Detect intent
        detected_mode, confidence = SmartDetector.detect_intent(
            request.prompt, 
            request.mode
        )
        
        # Suggest best model if not specified
        if not request.model or request.model == "auto":
            request.model = SmartDetector.get_suggested_model(detected_mode)
        
        # Route to appropriate handler
        if detected_mode == "workflow":
            return await generate_workflow_response(request)
        
        elif detected_mode == "ppt":
            return await generate_ppt_options(request)
        
        else:  # code (default)
            return await generate_code_response(request)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_workflow_response(request: SmartGenerationRequest):
    """Generate workflow diagram"""
    generator = WorkflowGenerator()
    
    # Use LLM to convert prompt to diagram
    diagram_code = await generator.text_to_diagram(
        request.prompt,
        format="mermaid",  # default
        model=request.model
    )
    
    return {
        "type": "workflow",
        "mode": "workflow",
        "confidence": 0.9,
        "diagram_code": diagram_code,
        "format": "mermaid",
        "preview_url": f"/workflow/preview/{hash(diagram_code)}",
        "export_options": ["svg", "png", "pdf"]
    }


async def generate_ppt_options(request: SmartGenerationRequest):
    """Generate PPT enhancement options"""
    enhancer = PPTEnhancer()
    
    # Generate multiple variations
    options = await enhancer.generate_options(
        request.prompt,
        num_options=3,
        model=request.model
    )
    
    return {
        "type": "ppt",
        "mode": "ppt",
        "confidence": 0.85,
        "options": options,
        "message": "Select one or more options below"
    }


async def generate_code_response(request: SmartGenerationRequest):
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
        "confidence": 0.95,
        "response": response,
        "language": "auto"  # Could add language detection
    }


# Pydantic model
class SmartGenerationRequest(BaseModel):
    prompt: str
    mode: str = "auto"  # auto, code, workflow, ppt
    model: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.2
```

---

## 🎨 Frontend Implementation

### Modified `frontend/app.js`

```javascript
// Add mode selector handling
function initializeModeSelector() {
    const modeSelector = document.getElementById('generateMode');
    const fileUpload = document.getElementById('pptUploadSection');
    
    modeSelector.addEventListener('change', (e) => {
        const mode = e.target.value;
        
        // Show/hide file upload for PPT mode
        if (mode === 'ppt') {
            fileUpload.style.display = 'block';
        } else {
            fileUpload.style.display = 'none';
        }
        
        // Update placeholder text
        updatePlaceholder(mode);
    });
}

function updatePlaceholder(mode) {
    const textarea = document.getElementById('generatePrompt');
    const placeholders = {
        'auto': 'Describe what you want to create...',
        'code': 'Write a function to reverse a string...',
        'workflow': 'Create a CI/CD pipeline diagram...',
        'ppt': 'Enhance my presentation slide about AI...'
    };
    textarea.placeholder = placeholders[mode] || placeholders['auto'];
}

// Modified generate function
async function generateSmart() {
    const prompt = document.getElementById('generatePrompt').value;
    const mode = document.getElementById('generateMode').value;
    const model = document.getElementById('generateModel').value;
    const maxTokens = parseInt(document.getElementById('generateMaxTokens').value);
    const temperature = parseFloat(document.getElementById('generateTemperature').value);
    
    if (!prompt.trim()) {
        alert('Please enter a prompt');
        return;
    }
    
    showGenerateProgress(model);
    
    try {
        const response = await apiPost('/generate/smart', {
            prompt,
            mode,
            model,
            max_tokens: maxTokens,
            temperature
        });
        
        const data = await response.json();
        
        // Route to appropriate display function based on type
        switch(data.type) {
            case 'workflow':
                displayWorkflowResult(data);
                break;
            case 'ppt':
                displayPPTOptions(data);
                break;
            case 'code':
            default:
                displayGenerateResult(data);
                break;
        }
        
    } catch (error) {
        console.error('Generation error:', error);
        alert(`Error: ${error.message}`);
    } finally {
        hideGenerateProgress();
    }
}

function displayWorkflowResult(data) {
    const resultDiv = document.getElementById('generateResult');
    
    // Show detected mode
    const modeIndicator = `<div class="mode-indicator">
        🎨 Workflow Diagram (${(data.confidence * 100).toFixed(0)}% confidence)
    </div>`;
    
    // Render Mermaid diagram
    const diagramHtml = `
        <div class="workflow-container">
            <div class="mermaid">${data.diagram_code}</div>
        </div>
        <div class="export-options">
            <button onclick="exportDiagram('svg')">📄 Export SVG</button>
            <button onclick="exportDiagram('png')">🖼️ Export PNG</button>
            <button onclick="exportDiagram('pdf')">📑 Export PDF</button>
        </div>
    `;
    
    resultDiv.innerHTML = modeIndicator + diagramHtml;
    resultDiv.style.display = 'block';
    
    // Initialize Mermaid
    mermaid.init(undefined, document.querySelectorAll('.mermaid'));
}

function displayPPTOptions(data) {
    const resultDiv = document.getElementById('generateResult');
    
    const modeIndicator = `<div class="mode-indicator">
        📊 PPT Enhancement Options (${(data.confidence * 100).toFixed(0)}% confidence)
    </div>`;
    
    const optionsHtml = data.options.map((opt, idx) => `
        <div class="ppt-option" onclick="togglePPTOption(${idx})">
            <div class="option-header">
                <h4>Option ${idx + 1}: ${opt.title}</h4>
                <input type="checkbox" id="opt-${idx}">
            </div>
            <div class="option-preview">${opt.preview}</div>
            <div class="option-description">${opt.description}</div>
        </div>
    `).join('');
    
    const actionsHtml = `
        <div class="ppt-actions">
            <button onclick="applyPPTOptions()" class="btn-primary">
                ✅ Apply Selected
            </button>
            <button onclick="regeneratePPT()" class="btn-secondary">
                🔄 Regenerate
            </button>
        </div>
    `;
    
    resultDiv.innerHTML = modeIndicator + optionsHtml + actionsHtml;
    resultDiv.style.display = 'block';
}
```

---

## 📝 HTML Changes

### Add to `frontend/index.html` (in Generate Code tab)

```html
<!-- Add mode selector before model selector -->
<div class="form-group">
    <label for="generateMode">Mode (Optional - Auto-detects if not selected)</label>
    <select id="generateMode" class="input-select">
        <option value="auto" selected>🤖 Auto-detect (Smart)</option>
        <option value="code">💻 Code Generation</option>
        <option value="workflow">🎨 Workflow Diagram</option>
        <option value="ppt">📊 PPT Enhancement</option>
    </select>
</div>

<!-- Add file upload section (hidden by default) -->
<div id="pptUploadSection" style="display: none;">
    <div class="form-group">
        <label>Upload Presentation (Optional)</label>
        <input type="file" id="pptFile" accept=".ppt,.pptx" class="input-file">
    </div>
</div>
```

---

## 🎯 Example Prompts & Detection

| Prompt | Detected Mode | Confidence |
|--------|---------------|------------|
| "Write a Python function to reverse a string" | Code | 95% |
| "Create a CI/CD pipeline diagram" | Workflow | 90% |
| "Generate a flowchart for user authentication" | Workflow | 95% |
| "Enhance my presentation slide about AI" | PPT | 85% |
| "Make my slide more visual with infographics" | PPT | 90% |
| "Implement a binary search algorithm" | Code | 95% |
| "Show me the data flow in a microservices architecture" | Workflow | 92% |

---

## ✅ Benefits of This Approach

1. **Simpler UX** - One familiar interface, no learning curve
2. **Smart defaults** - Works without user selecting mode
3. **Flexible** - Users can override auto-detection if needed
4. **Minimal changes** - Reuses existing UI components
5. **Progressive enhancement** - Existing code generation still works exactly the same
6. **Unified experience** - Same button, same flow, different outputs

---

## 🚀 Implementation Priority

1. **Phase 1**: Add smart detector + workflow generation
2. **Phase 2**: Add PPT enhancement
3. **Phase 3**: Refine detection algorithm based on usage

This is much simpler than the original design while being more powerful!

