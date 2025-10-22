# Semantic Search CLI

A simple command-line tool for crawling directories, converting documents to markdown, and performing semantic search using hybrid dense/sparse embeddings with AI-powered responses.

## Features

- **Document Processing**: Automatically converts PDF, Excel, Word, and PowerPoint files to markdown
- **Hybrid Search**: Combines dense and sparse embeddings for superior search accuracy
- **AI Integration**: Uses OpenAI-compatible APIs (including Google Gemini) for intelligent responses
- **Vector Database**: Built on Qdrant for efficient similarity search
- **Modern CLI**: Beautiful terminal interface with progress bars, colored output, and interactive elements
- **Incremental Processing**: Filter files by modification time for efficient updates

## Supported File Types

- PDF (`.pdf`)
- Microsoft Excel (`.xlsx`)
- Microsoft Word (`.doc`, `.docx`)
- Microsoft PowerPoint (`.ppt`, `.pptx`)

## Prerequisites

- Rust 1.70+
- Qdrant vector database running locally or remotely
- OpenAI API key (or compatible API like Google Gemini)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd semantic-search-cli
```

2. Install dependencies:

```bash
cargo build --release
```

3. Set up environment variables:

```bash
cp env.example .env
# Edit .env with your API keys and configuration
```

## Configuration

Create a `.env` file with the following variables:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
OPENAI_MODEL=gemini-2.5-flash

# Qdrant Configuration
QDRANT_URL=http://localhost:6334
```

### API Configuration Options

- **OpenAI**: Use `https://api.openai.com/v1/chat/completions` with `gpt-3.5-turbo` or `gpt-4`
- **Google Gemini**: Use `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions` with `gemini-2.5-flash`
- **Other OpenAI-compatible APIs**: Configure URL and model as needed

## Usage

### Basic Document Processing

Process all supported files in a directory:

```bash
cargo run -- --directory data
```

### Search Documents

Perform semantic search across processed documents:

```bash
cargo run -- --search "What are the safety requirements for tower installation?"
```

### Filter by Modification Time

Only process files modified since a specific timestamp:

```bash
cargo run -- --directory data --since 1640995200  # Unix timestamp
```

### Command Line Options

- `--directory, -d`: Directory to crawl (default: `data`)
- `--since, -s`: Only process files modified since this Unix timestamp
- `--embed`: Convert files to markdown and print content
- `--search`: Perform semantic search with the given query

## Architecture

### Components

1. **Document Converter**: Uses `markitdown` to convert various file formats to markdown
2. **Text Chunker**: Splits markdown content into semantic chunks using `text-splitter`
3. **Embedding Generator**: Creates both dense and sparse embeddings using `fastembed`
4. **Vector Store**: Stores embeddings and metadata in Qdrant
5. **Search Engine**: Performs hybrid search combining dense and sparse vectors
6. **AI Client**: Generates intelligent responses using OpenAI-compatible APIs

### Embedding Models

- **Dense**: BGESmallENV15 (384 dimensions)
- **Sparse**: SPLADEPPV1
- **Reranker**: JINARerankerV1TurboEn

### Search Process

1. Generate dense and sparse embeddings for the query
2. Perform vector similarity search in Qdrant
3. Combine results using Reciprocal Rank Fusion (RRF)
4. Rerank top results using the reranker model
5. Generate AI response based on retrieved context

## Project Structure

```
src/
├── main.rs           # Main application logic and CLI interface
├── cli_ui.rs         # Terminal UI components and styling
├── ai.rs             # OpenAI-compatible API client
└── qdrant_client.rs  # Qdrant vector database operations
```

## Troubleshooting

### Qdrant Connection Issues

Ensure Qdrant is running:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### API Key Issues

Verify your API key is correctly set in the `.env` file and has sufficient credits.

### File Conversion Errors

Some files may fail to convert due to:

- Corrupted file format
- Password protection
- Unsupported content

The tool will skip failed conversions and continue processing other files.

## Releases

```bash
# Tag release
git tag -a 0.2.0 -m "Message"

# Push tag and CI does the rest
git push --tags
```
