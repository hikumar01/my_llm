# Feature Design: Workflow Generation & PPT Enhancement

## 🎯 Overview

This document outlines the design for two major new features:
1. **Workflow Generator**: Create visual workflows/diagrams from text descriptions
2. **PPT Enhancer**: Upload presentations and generate multiple enhancement options with interactive selection

---

## 📊 Architecture Summary

### New Components

#### Backend (Python/FastAPI)
- `src/workflow_generator.py` - Text-to-diagram conversion
- `src/ppt_processor.py` - PPTX parsing and manipulation
- `src/ppt_enhancer.py` - Multi-option generation engine
- `src/image_utils.py` - Image processing utilities
- `src/cache_manager.py` - Session and option caching

#### Frontend (JavaScript)
- `frontend/workflow_tab.js` - Workflow generation UI
- `frontend/ppt_enhancer_tab.js` - PPT upload and enhancement UI
- `frontend/preview_panel.js` - Side-by-side comparison view
- `frontend/option_selector.js` - Multi-option selection logic

#### New API Endpoints
```
Workflow:
  POST   /workflow/generate       - Generate workflow from text
  POST   /workflow/stream         - Stream workflow generation
  GET    /workflow/export/:id     - Export in various formats

PPT Enhancement:
  POST   /ppt/upload              - Upload PPT file
  GET    /ppt/slides/:id          - Get slide thumbnails
  POST   /ppt/generate-options    - Generate enhancement variations
  POST   /ppt/apply-option        - Apply selected options
  GET    /ppt/preview/:id         - Preview enhanced slide
  GET    /ppt/download/:id        - Download enhanced PPT
```

---

## 🎨 Feature 1: Workflow Generator

### User Flow
1. User enters workflow description (e.g., "CI/CD pipeline with testing stages")
2. Selects format: Mermaid, GraphViz, Flowchart, Sequence Diagram
3. Chooses style: Professional, Creative, Technical
4. Clicks "Generate Workflow"
5. Views interactive preview (SVG/Canvas)
6. Exports as PNG, SVG, PDF, or embed code

### Technical Implementation

#### Input Processing
```python
# workflow_generator.py
def generate_workflow(description: str, format: str, style: str):
    # 1. Use LLM to convert text to diagram syntax
    prompt = f"""Convert this description to {format} diagram:
    {description}
    
    Style: {style}
    Output only the diagram code, no explanations."""
    
    # 2. Generate diagram code
    diagram_code = llm_client.generate(prompt)
    
    # 3. Render to visual format
    if format == "mermaid":
        svg = render_mermaid(diagram_code)
    elif format == "graphviz":
        svg = render_graphviz(diagram_code)
    
    # 4. Convert to other formats if needed
    png = svg_to_png(svg)
    
    return {
        "code": diagram_code,
        "svg": svg,
        "png": png,
        "format": format
    }
```

#### Rendering Options
- **Mermaid**: Client-side rendering using mermaid.js
- **GraphViz**: Server-side using graphviz-wasm or pygraphviz
- **Export**: Convert SVG to PNG/PDF using Pillow/ReportLab

### Models to Use
- **Primary**: `llama3.2:3b` (fast, good for structured output)
- **Alternative**: `qwen2.5-coder:7b` (better for complex diagrams)

---

## 📊 Feature 2: PPT Enhancement with Multi-Option Selection

### User Flow
1. Upload PPT file (drag & drop or browse)
2. View slide thumbnails in grid
3. Select one or more slides to enhance
4. Enter enhancement request (e.g., "Make slide 2 more visual with infographics")
5. Choose number of variations (3-5)
6. Click "Generate Enhancement Options"
7. **Review multiple options side-by-side**
8. **Select preferred options** (can select multiple)
9. **Compare selected options** in detail
10. Request modifications or regenerate
11. Apply selected enhancements
12. Download enhanced PPT or continue iterating

### Key Innovation: Multi-Option Selection

#### Why Multiple Options?
- Users can see different creative approaches
- Mix and match elements from different options
- Iterative refinement with context
- Better user control and satisfaction

#### Selection Interface
```
┌─────────────────────────────────────────────────────┐
│ Original Slide 2                                    │
│ [Current content preview]                           │
└─────────────────────────────────────────────────────┘

┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Option 1 │  │ Option 2 │  │ Option 3 │  │ Option 4 │
│ Modern   │  │Minimalist│  │Infograph.│  │Data-Focus│
│ ✓ SELECT │  │ ☐ Select │  │ ✓ SELECT │  │ ☐ Select │
│ ⭐⭐⭐⭐⭐ │  │ ⭐⭐⭐⭐☆ │  │ ⭐⭐⭐⭐⭐ │  │ ⭐⭐⭐☆☆ │
└──────────┘  └──────────┘  └──────────┘  └──────────┘

┌─────────────────────────────────────────────────────┐
│ Side-by-Side Comparison                             │
│ [Original] [Option 1] [Option 3]                    │
│ Toggle: Grid | Slider | Overlay                     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Modification Request:                               │
│ "Combine Option 1's layout with Option 3's icons"  │
└─────────────────────────────────────────────────────┘

[✅ Apply Selected (2)] [🔄 Regenerate] [✏️ Modify] [💾 Download]
```

### Technical Implementation

#### 1. PPT Upload & Parsing
```python
# ppt_processor.py
from pptx import Presentation
from PIL import Image

def parse_ppt(file_path: str):
    prs = Presentation(file_path)
    slides = []
    
    for idx, slide in enumerate(prs.slides):
        # Extract text content
        text_content = extract_text(slide)
        
        # Generate thumbnail
        thumbnail = generate_slide_thumbnail(slide, idx)
        
        # Extract layout info
        layout_info = analyze_layout(slide)
        
        slides.append({
            "index": idx,
            "text": text_content,
            "thumbnail": thumbnail,
            "layout": layout_info
        })
    
    return {
        "total_slides": len(slides),
        "slides": slides,
        "metadata": extract_metadata(prs)
    }
```

#### 2. Multi-Option Generation
```python
# ppt_enhancer.py
def generate_enhancement_options(
    slide_content: dict,
    enhancement_request: str,
    num_options: int = 4,
    style: str = "professional"
):
    options = []
    
    # Generate diverse variations
    variation_prompts = [
        "modern design with bold colors and clean layout",
        "minimalist approach with lots of white space",
        "infographic style with visual icons and diagrams",
        "data-focused with charts and metrics",
        "creative design with unique visual elements"
    ]
    
    for i in range(num_options):
        prompt = f"""
        Original slide content:
        {slide_content['text']}
        
        Enhancement request: {enhancement_request}
        
        Variation style: {variation_prompts[i]}
        Overall style: {style}
        
        Generate enhanced slide content with:
        1. Improved layout structure
        2. Visual elements to add
        3. Color scheme suggestions
        4. Typography recommendations
        5. Content reorganization
        
        Output as JSON with specific instructions.
        """
        
        # Generate enhancement
        enhancement = llm_client.generate(prompt)
        
        # Render preview
        preview = render_slide_preview(slide_content, enhancement)
        
        options.append({
            "id": f"opt_{i}",
            "title": variation_prompts[i].split()[0].title(),
            "enhancement": enhancement,
            "preview": preview,
            "changes": extract_changes(enhancement)
        })
    
    # Cache options for session
    cache_manager.store_options(session_id, options)
    
    return options
```

#### 3. Option Selection & Comparison
```javascript
// frontend/option_selector.js
class OptionSelector {
    constructor() {
        this.selectedOptions = new Set();
        this.comparisonView = null;
    }
    
    toggleOption(optionId) {
        if (this.selectedOptions.has(optionId)) {
            this.selectedOptions.delete(optionId);
        } else {
            this.selectedOptions.add(optionId);
        }
        this.updateUI();
        this.updateComparison();
    }
    
    updateComparison() {
        const selected = Array.from(this.selectedOptions);
        if (selected.length > 0) {
            this.showSideBySide(['original', ...selected]);
        }
    }
    
    showSideBySide(optionIds) {
        const container = document.getElementById('comparison-view');
        container.innerHTML = '';
        
        optionIds.forEach(id => {
            const preview = this.createPreviewPanel(id);
            container.appendChild(preview);
        });
        
        // Enable slider/overlay controls
        this.enableComparisonControls();
    }
    
    async applySelected() {
        const selected = Array.from(this.selectedOptions);
        const response = await fetch('/ppt/apply-option', {
            method: 'POST',
            body: JSON.stringify({
                slide_id: this.currentSlide,
                option_ids: selected,
                merge_strategy: 'best_of_each'
            })
        });
        
        const result = await response.json();
        this.showFinalPreview(result.enhanced_slide);
    }
}
```

#### 4. Applying Enhancements
```python
# ppt_processor.py
def apply_enhancements(
    original_ppt: Presentation,
    slide_index: int,
    selected_options: List[str],
    merge_strategy: str = "best_of_each"
):
    slide = original_ppt.slides[slide_index]
    
    # Get cached enhancement options
    options = cache_manager.get_options(session_id, selected_options)
    
    if merge_strategy == "best_of_each":
        # Intelligently merge multiple options
        merged = merge_enhancements(options)
    else:
        # Use single option
        merged = options[0]
    
    # Apply layout changes
    apply_layout(slide, merged['layout'])
    
    # Apply visual elements
    add_visual_elements(slide, merged['visuals'])
    
    # Apply color scheme
    apply_colors(slide, merged['colors'])
    
    # Update content
    update_content(slide, merged['content'])
    
    return original_ppt
```

---

## 🔧 Required Dependencies

### Python Packages
```txt
python-pptx==0.6.23          # PPT manipulation
Pillow==10.1.0               # Image processing
graphviz==0.20.1             # GraphViz rendering
reportlab==4.0.7             # PDF generation
```

### JavaScript Libraries
```html
<!-- Mermaid for diagram rendering -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>

<!-- Image comparison slider -->
<script src="https://cdn.jsdelivr.net/npm/img-comparison-slider@8/dist/index.js"></script>
```

### New LLM Models
```bash
# Vision-capable models for visual understanding
ollama pull llama3.2-vision:11b

# Vision-language model
ollama pull qwen2-vl:7b

# General purpose for text enhancement
ollama pull mistral:7b
```

---

## 📁 File Structure

```
my_llm/
├── src/
│   ├── workflow_generator.py      # NEW
│   ├── ppt_processor.py           # NEW
│   ├── ppt_enhancer.py            # NEW
│   ├── image_utils.py             # NEW
│   ├── cache_manager.py           # NEW
│   └── api_server.py              # MODIFIED
├── frontend/
│   ├── workflow_tab.js            # NEW
│   ├── ppt_enhancer_tab.js        # NEW
│   ├── preview_panel.js           # NEW
│   ├── option_selector.js         # NEW
│   ├── index.html                 # MODIFIED
│   ├── styles.css                 # MODIFIED
│   └── app.js                     # MODIFIED
├── uploads/                       # NEW - temporary file storage
├── cache/                         # NEW - option caching
└── exports/                       # NEW - generated files
```

---

## 🚀 Implementation Phases

### Phase 1: Workflow Generator (Week 1-2)
- [ ] Add workflow models to constants
- [ ] Implement workflow_generator.py
- [ ] Create workflow tab UI
- [ ] Add Mermaid rendering
- [ ] Test with various diagram types

### Phase 2: PPT Upload & Parsing (Week 3)
- [ ] Implement ppt_processor.py
- [ ] Create upload endpoint
- [ ] Build slide thumbnail generator
- [ ] Create PPT enhancer tab UI

### Phase 3: Multi-Option Generation (Week 4-5)
- [ ] Implement ppt_enhancer.py
- [ ] Build option generation logic
- [ ] Create preview rendering
- [ ] Implement caching system

### Phase 4: Interactive Selection (Week 6)
- [ ] Build option selector component
- [ ] Implement comparison view
- [ ] Add modification request flow
- [ ] Create apply/download logic

### Phase 5: Testing & Polish (Week 7)
- [ ] End-to-end testing
- [ ] UI/UX refinements
- [ ] Performance optimization
- [ ] Documentation

---

## 💡 Key Features Summary

### Workflow Generator
✅ Text-to-diagram conversion
✅ Multiple format support (Mermaid, GraphViz, etc.)
✅ Interactive preview
✅ Export to PNG/SVG/PDF
✅ Streaming generation

### PPT Enhancer
✅ Drag & drop PPT upload
✅ Slide thumbnail grid
✅ Multi-slide selection
✅ **Generate 3-5 enhancement options**
✅ **Interactive option selection**
✅ **Side-by-side comparison**
✅ **Iterative refinement**
✅ Apply selected enhancements
✅ Download enhanced PPT

---

## 🎯 Next Steps

1. Review this design document
2. Confirm feature requirements
3. Begin Phase 1 implementation
4. Set up development environment with new dependencies
5. Create initial prototypes for user feedback

Would you like me to start implementing any specific phase?

