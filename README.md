# Supplementary Material for the Paper:
## "Aligning AI Model’s Knowledge and Conceptual Model’s Symbols"

This repository contains all supplementary resources for the ALeKS framework, as described in the paper. It includes datasets, processing scripts, evaluation workflows, a web-based visualization tool, and serialized embeddings to support reproducibility and further research.

---

## 🔖 Repository Structure

```
mm_cm_alignment/
├── README                         # This file
├── datasets/                     # Input datasets and multimedia content
├── VisualizerTool_Web/          # Web-based visual interface for inspecting embeddings
│   ├── index                     # Main PHP file for launching the tool
│   └── logo                      # Logo image used in the interface
├── processing_eval_A/           # Evaluation scripts and embeddings for OntoUML analysis
│   ├── *.py                      # Processing and alignment pipeline scripts
│   ├── *_embeddings_*.csv       # 2D/3D embeddings for visualization and learning
│   └── trained_models/          # Saved PyTorch MLP models
├── cross_lang_eval_B/           # Evaluation across modeling languages (UML, BPMN, SysML, EA, etc.)
│   ├── *.py                      # Cross-language extraction and embedding scripts
│   ├── extracted_*.json         # Serialized models (JSON format)
│   └── model_embeddings_*.csv   # Aligned embeddings for each modeling language
```

---

## 📁 Folders Explained

### `datasets/`
- Placeholder directory for all modeling datasets.

### `VisualizerTool_Web/`
- Contains a lightweight PHP-based front-end to visualize embedding alignments between NLT, CMT, and multimodal concepts.
- `index` is the main entry point for browsing and comparing serialized models.
- `logo` provides branding for the tool interface.

### `processing_eval_A/`
- Full OntoUML serialization and alignment pipeline.
- Includes:
  - Scripts for collecting and serializing models.
  - Training MLPs to learn semantic shift between NLT → CMT.
  - Dimensionality reduction (t-SNE/PCA) and visualization.
  - Summarization and AI comparison routines.
- Output files:
  - `ontouml_embeddings_*.csv` for aligned and reduced embeddings.
  - `ontouml_models_*.json/csv` for serialized model structures.
  - `trained_models/` for saved PyTorch alignment models.

### `cross_lang_eval_B/`
- Evaluates generalizability across six modeling repositories:
  - OntoUML, EA Model Set, BPMN (research and HD), SysML PhS, ModelSet.
- Includes:
  - Scripts for model extraction and alignment.
  - Multimodal embedding integration and visualization.
  - Serialized JSON models and aligned embeddings in CSV format.

---

## 🧪 How to Reproduce the Evaluation

1. Install dependencies

2. Run the OntoUML processing pipeline:
   ```bash
   cd processing_eval_A
   python 1_collect_ontouml_models.py
   python 2_serialize_cmt_ontouml_models.py
   ...
   ```

3. Train and visualize alignment:
   ```bash
   python 7_train_2d_ontouml_alignment.py
   python 6_reduce_dim_and_visualize_ontouml_embeddings.py
   ```

4. Explore cross-language embeddings:
   ```bash
   cd ../cross_lang_eval_B
   python e1_onto.py
   python e2_modelset.py
   ...
   ```

5. Launch the visual tool (requires PHP server):
   ```bash
   cd ../VisualizerTool_Web
   php -S localhost:8080
   ```
