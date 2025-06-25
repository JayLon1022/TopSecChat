# <img src="assets/tianrongxin.png" width="40px" align="center" alt="TopSecChat logo"> TopSecChat

A modular AI assistant for secure enterprise Q&A, built with LlamaIndex and LLM APIs.

**Intern Project in TopSec.**

---

## 📋 Table of Contents

1. [🗂️ Project Structure](#️-project-structure)
2. [🚀 Features](#-features)
3. [📦 Installation](#-installation)
   - [Install from PyPI](#install-from-pypi)
   - [Install from Source](#install-from-source)
4. [📘 Usage Instructions](#-usage-instructions)
5. [📝 License](#-license)
6. [🛠 Contributing](#-contributing)

---

## 🗂️ Project Structure

```
.
├── app.py                        # FastAPI backend main program
├── client_test.py                # Command-line test client
├── upload_database.py            # Knowledge base vectorization & import script
├── conf.yaml                     # Configuration file
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment dependencies
├── assets/                       # Static assets (e.g., logo)
├── crawler/                      # Crawlers & text processing scripts
│   ├── rename.py
│   ├── pdf_to_text.py
│   ├── text_polish.py
│   └── crawl.ipynb
├── storage/                      # Vector DB and storage
│   ├── topsec_document_knowledge/
│   └── security_knowledge/
├── topsec_pdf/                   # TopSec-related original PDF docs
├── topsec_document_knowledge/    # TopSec PDF converted txt docs
├── security_pdf/                 # General security original PDF docs
├── security_knowledge/           # General security PDF converted txt docs
└── .gitignore
```

---

## 🚀 Features

- Modular RAG-based Q&A system for enterprise security knowledge
- Supports both TopSec and general security document ingestion
- Automatic PDF-to-text conversion and vector database update
- FastAPI backend for API service
- Command-line client for quick testing

---

## 📦 Installation

### [TBD] Install from PyPI

```bash
pip install tschat
```

### Install from source

```bash
git clone https://github.com/Y0oMu/RAG_topsec.git
cd TopSecChat
pip install -r requirements.txt
```

---

## 📘 Usage Instructions

### 1. Prepare the Knowledge Base

Place your PDF documents into the corresponding folders:
- `topsec_pdf/` for TopSec-related documents
- `security_pdf/` for general security documents

The system will automatically convert PDFs to `.txt` and store them in `topsec_document_knowledge/` and `security_knowledge/`.

### 2. Build or Update the Vector Database

To automatically add, update, or remove documents in the RAG vector store:

```bash
python upload_database.py
```

This script will parse all PDFs in `topsec_pdf/` and `security_pdf/`, convert them to text, and update the vector database under the `storage/` directory accordingly.

### 3. Run the Backend Service

Start the backend server by running:

```bash
python app.py
```

This will launch the FastAPI-based API service that handles retrieval, generation, and response delivery.

Run the command-line chat client for testing:

```bash
python client_test.py
```

---

## 📝 [TBD] License

[MIT License](LICENSE)

---

## 🛠 Contributing

We welcome contributions to improve TopSecChat! Please follow these steps:

1. Fork the repository
2. Create a new branch (`feature-branch`)
3. Commit your changes
4. Open a pull request
