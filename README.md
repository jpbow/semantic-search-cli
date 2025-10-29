# Semantic Search CLI

A simple command-line tool for crawling directories, converting documents to markdown, and performing semantic search using hybrid dense/sparse embeddings with AI-powered responses.

## Supported File Types

- PDF (`.pdf`)
- Microsoft Excel (`.xlsx`)
- Microsoft Word (`.doc`, `.docx`)
- Microsoft PowerPoint (`.ppt`, `.pptx`)

## Project Structure

```
src/
├── main.rs           # Main application logic and CLI interface
├── cli_ui.rs         # Terminal UI components and styling
├── ai.rs             # OpenAI-compatible API client
└── qdrant_client.rs  # Qdrant vector database operations
```

## Installation & Quick Start

### Prerequisites

- Docker
- Rust 1.70+
- OpenAI API key (or compatible API like Google Gemini)

1. Install required system packages on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y libssl-dev pkg-config
```

2. Clone the repository:

```bash
git clone <repository-url>
cd semantic-search-cli
```

3. Install dependencies:

```bash
cargo build --release
```

4. Set up environment variables:

```bash
cp env.example .env
# Edit .env with your API keys and configuration
```

AI config

- **OpenAI**: Use `https://api.openai.com/v1/chat/completions` with `gpt-3.5-turbo` or `gpt-4`
- **Google Gemini**: Use `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions` with `gemini-2.5-flash`
- **Other OpenAI-compatible APIs**: Configure URL and model as needed

5. Start the Qdrant container

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

6. Run!

```bash
# Process all supported files in a directory:
cargo run -- --directory data

# Perform semantic search across processed documents:
cargo run -- --search "What are the safety requirements for tower installation?"

# Only process files modified since a specific timestamp:
cargo run -- --directory data --since 1640995200  # Unix timestamp
```

### Command Line Options

- `--directory, -d`: Directory to crawl (default: `data`)
- `--since, -s`: Only process files modified since this Unix timestamp
- `--embed`: Convert files to markdown and print content
- `--search`: Perform semantic search with the given query

### Embedding Models Used

- **Dense**: BGESmallENV15 (384 dimensions)
- **Sparse**: SPLADEPPV1
- **Reranker**: JINARerankerV1TurboEn

### Search Process

1. Generate dense and sparse embeddings for the query
2. Perform vector similarity search in Qdrant
3. Combine results using Reciprocal Rank Fusion (RRF)
4. Rerank top results using the reranker model
5. Generate AI response based on retrieved context

## Releases

```bash
# Tag release
git tag -a 0.2.0 -m "Message"

# Push tag and CI does the rest
git push --tags
```
