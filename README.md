# Argus: Can Argus Judge Them All? Comparing VLMs Across Domains

Argus is a benchmarking framework for evaluating **Vision-Language Models (VLMs)**—notably **CLIP**, **BLIP**, and **LXMERT**—across a diverse set of multimodal tasks and datasets. Drawing inspiration from the all-seeing giant of Greek mythology, Argus systematically examines the strengths, weaknesses, and generalization capabilities of leading VLMs, with a special focus on **cross-domain consistency, efficiency, and real-world deployment** in resource-constrained environments.

---

## 📚 Supported Datasets

Argus benchmarks VLMs on five representative datasets, each targeting different aspects of multimodal understanding:

- **COCO**  
  Large-scale dataset for object detection, segmentation, and captioning.

- **Flickr30k**  
  Image captioning dataset with diverse, real-world scenes.

- **CLEVR**  
  Synthetic dataset designed for compositional language and visual reasoning.

- **VCR**  
  Visual Commonsense Reasoning dataset for complex image-based QA and rationale selection.

- **Visual Genome**  
  Dataset with dense object, attribute, and relationship annotations for fine-grained scene understanding.

Prepare each dataset in the `data/` directory as specified in the provided notebooks.

---

## 🧠 Vision-Language Models Benchmarked

Argus evaluates the following state-of-the-art VLMs:

| Model   | Core Strengths                                            | Key Weaknesses                |
|---------|----------------------------------------------------------|-------------------------------|
| **CLIP**    | Strong generalization, robust cross-domain consistency | Weaker in generation and structured reasoning |
| **BLIP**    | Excels at curated data, strong captioning and retrieval | Variable on less curated domains |
| **LXMERT**  | Superior in structured visual reasoning tasks         | Less adaptable, higher resource use |

- **CLIP**: Contrastive Language–Image Pretraining; excels at zero-shot generalization via contrastive learning on web-scale image-text pairs.
- **BLIP**: Bootstrapped Language Image Pretraining; uses a Mixture-of-Experts (MED) encoder-decoder architecture for unified understanding and generation, leveraging synthetic caption bootstrapping and multi-task objectives.
- **LXMERT**: Transformer-based model with cross-modal attention, optimized for structured reasoning and VQA.

---

## 🛠️ Usage

### 1. Installation

Install dependencies:

pip install -r requirements.txt

**Core dependencies:**  
- Python 3.8+  
- PyTorch  
- Transformers  
- OpenAI CLIP  
- Additional packages as listed in `requirements.txt`

---

### 2. Embedding Extraction

Extract image-text embeddings for your dataset:

python blip_embedding_extractor.py --input data/coco/ --output embeddings/blip_coco.pkl
python clip_embedding_extractor.py --input data/coco/ --output embeddings/clip_coco.pkl

---

### 3. Training and Evaluation

Use the provided Jupyter notebooks for model training and evaluation:

- `blip-model.py` — BLIP experiments
- `clip-model.py` — CLIP experiments
- `vcr-blip-model.py` — BLIP on VCR dataset

Custom workload prediction wrappers:

- `Clever_blip_model.py` — BLIP-based workload prediction
- `Clever_clip_model.py` — CLIP-based workload prediction

---

## 🧩 File Structure

| File/Folder                | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| `Clever_clip&blip/`       | CLEVR dataset experiments with CLIP & BLIP                   |
| `Coco_vlip&clip/`         | COCO dataset experiments with CLIP & CLIP                    |
| `Flickr30k_clip&blip/`    | Flickr30k dataset with CLIP & BLIP                           |
| `VCR_clip&blip/`          | VCR dataset experiments with CLIP & BLIP                     |
| `vgenome_blip&clip/`      | Visual Genome dataset with BLIP & CLIP                       |
| `README.md`               | Project documentation                                        |

---

## ⚡ Experiments & Results

### Key Evaluation Dimensions

- **Prediction Accuracy** (retrieval, captioning, reasoning)
- **Generation Quality** (BLEU, METEOR, CIDEr, SPICE)
- **Computational Efficiency** (latency, memory, FLOPs, throughput)
- **Generalization** using the novel **Cross-Dataset Consistency (CDC)** metric.

### Main Findings

| Task / Metric                 | BLIP           | CLIP           | LXMERT         |
|-------------------------------|----------------|----------------|----------------|
| **Image-Text Retrieval**      | Best on COCO, Flickr30k | Strong generalization | Best on Visual Genome |
| **Caption Generation**        | Highest BLEU, METEOR, CIDEr | Weakest | Competitive BLEU, lower diversity |
| **Visual Reasoning (CLEVR)**  | 89.7%          | 84.5%          | **96.3%**      |
| **Commonsense QA (VCR)**      | 74.8% (Q→A)    | 62.3% (Q→A)    | 70.2% (Q→A), **71.5% (QA→R)** |
| **Efficiency (COCO, latency)**| 120ms          | **30ms**       | 150ms          |
| **CDC (Generalization)**      | 0.76           | **0.92**       | 0.64           |

- **BLIP**: Outperforms in captioning and retrieval, especially on curated datasets.
- **CLIP**: Most robust across domains, with top CDC score (0.92), making it ideal for general-purpose and real-world deployment.
- **LXMERT**: Excels in structured reasoning and VQA, but less efficient and less generalizable.

---

## 🧮 Cross-Dataset Consistency (CDC) Metric

**Cross-Dataset Consistency (CDC)** measures how reliably a model maintains its performance across multiple datasets.

$$\text{CDC}(M_j) = 1 - \frac{1}{|D|} \sum_{i=1}^{|D|} \left| \frac{a_{i,j} - \bar{a}_j}{\bar{a}_j} \right|$$

**Where:**
- $$\( M_j \)$$: The model being evaluated
- $$\( D \)$$: The set of datasets
- $$\( a_{i,j} \)$$: The accuracy (or relevant metric) of model \( M_j \) on dataset \( i \)
- $$\( \bar{a}_j \)$$: The mean performance of \( M_j \) across all datasets

**Interpretation:**
- **CDC = 1:** The model's performance is perfectly consistent across all datasets.
- **Lower CDC values:** Indicate greater performance fluctuation between datasets.

**Industrial relevance:**  
CDC highlights and rewards models that deliver *balanced* performance, which is essential for building robust, fair, and reproducible AI systems.



## 🏁 Benchmarking Protocol

- Unified preprocessing (image resizing, text tokenization, normalization)
- Batch inference with consistent hardware (NVIDIA A100 GPU, batch size 32)
- Multiple random splits for statistical rigor
- No test-time augmentation or ensembling.

---

## 🏆 Summary Table: VLMs Across Domains

| Model   | Retrieval | Captioning | Reasoning | Efficiency | Generalization (CDC) |
|---------|-----------|------------|-----------|------------|----------------------|
| BLIP    | ★★★★☆    | ★★★★★     | ★★★☆☆    | ★★☆☆☆     | ★★★☆☆               |
| CLIP    | ★★★★☆    | ★★☆☆☆     | ★★☆☆☆    | ★★★★★     | ★★★★★               |
| LXMERT  | ★★★☆☆    | ★★★☆☆     | ★★★★★    | ★☆☆☆☆     | ★★☆☆☆               |

---

## ⚠️ Limitations

- Benchmarks may not capture real-world diversity, noise, or domain-specific constraints.
- No task-specific fine-tuning; results reflect fixed checkpoints.
- Evaluation on edge devices is limited to theoretical analysis due to hardware constraints.

---

## 🤝 Contributing

Contributions are welcome!  
Open issues or submit pull requests for improvements, bug fixes, or new features.

---

## 📄 License

MIT License

---

**Contact:**  
For questions or collaboration, please open an issue or contact the maintainer.

---


