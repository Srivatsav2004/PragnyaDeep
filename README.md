# PragnyaDeep

Sandhi in Sanskrit involves complex phonetic transformations when words or morphemes 
combine, making it challenging to split words correctly. This poses difficulties for learners, 
researchers, and NLP applications.   
 
Our AI model addresses this by leveraging Retrieval-Augmented Generation (RAG), machine 
learning, and linguistic rules to accurately identify and split Sandhi in Sanskrit. The RAG model 
enhances accuracy by retrieving relevant linguistic data before generating precise splits, 
ensuring both context-aware and rule-based segmentation.
---

## Table of Contents

- [Introduction](#introduction)  
- [Features](#features)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
- [Usage](#usage)  
- [sandhi_samples_v2.txt](#sandhi_samples_v2txt)  
- [Architecture & Approach](#architecture--approach)  
- [Contributing](#contributing)  
- [License](#license)

---

## Introduction

Sandhi—the fusion of sounds in Sanskrit—presents a complex challenge in word segmentation.  
**PragnyaDeep** leverages a **Streamlit-powered interface** and a **RAG framework** enriched with linguistic rule sets to accurately and contextually split Sandhi, overcoming limitations of purely statistical or rule-based systems.

---

## Features

- 🎯 **Accurate Sandhi Splitting** – combines AI and linguistic rules  
- 🔍 **Context-Aware** – uses retrieval to enhance prediction  
- 🖥️ **Streamlit UI** – easy to test and visualize results  
- 📂 **Sample Dataset** – includes `sandhi_samples_v2.txt` for testing  

---

## Getting Started

### Prerequisites

- Python 3.7+  
- Streamlit installed  
- Recommended: Virtual environment  

### Installation

```bash
# Clone the repository
git clone https://github.com/Srivatsav2004/PragnyaDeep.git
cd PragnyaDeep

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
