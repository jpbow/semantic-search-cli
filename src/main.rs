mod ai;
mod cli_ui;
mod qdrant_client;

use clap::Parser;
use cli_ui::{CliUI, FileInfo};
use colored::Colorize;
use fastembed::{
    EmbeddingModel, InitOptions, RerankInitOptions, RerankerModel, SparseModel,
    SparseTextEmbedding, TextEmbedding, TextRerank,
};
use markitdown::MarkItDown;
use regex::Regex;
use std::env;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use text_splitter::MarkdownSplitter;
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "file-crawler")]
#[command(
    about = "A CLI tool that crawls a directory, converts files to markdown, and semantically chunks the content"
)]
struct Args {
    /// Directory to crawl (defaults to 'data')
    #[arg(short, long, default_value = "data")]
    directory: String,

    /// Only show files modified since this timestamp (Unix timestamp)
    #[arg(short, long)]
    since: Option<u64>,

    /// Convert files to markdown and print content
    #[arg(long)]
    embed: bool,

    /// Search query for semantic search
    #[arg(long)]
    search: Option<String>,
}

fn format_markdown(markdown: &str) -> String {
    let mut result = markdown.to_string();

    // Bold
    let bold_regex = Regex::new(r"\*\*(.*?)\*\*").unwrap();
    result = bold_regex
        .replace_all(&result, |caps: &regex::Captures| {
            format!("\x1b[1m{}\x1b[0m", &caps[1])
        })
        .to_string();

    // Italic
    let italic_regex = Regex::new(r"\*(.*?)\*").unwrap();
    result = italic_regex
        .replace_all(&result, |caps: &regex::Captures| {
            format!("\x1b[3m{}\x1b[0m", &caps[1])
        })
        .to_string();

    // Underline
    let underline_regex = Regex::new(r"__(.*?)__").unwrap();
    result = underline_regex
        .replace_all(&result, |caps: &regex::Captures| {
            format!("\x1b[4m{}\x1b[0m", &caps[1])
        })
        .to_string();

    // Strikethrough
    let strikethrough_regex = Regex::new(r"~~(.*?)~~").unwrap();
    result = strikethrough_regex
        .replace_all(&result, |caps: &regex::Captures| {
            format!("\x1b[9m{}\x1b[0m", &caps[1])
        })
        .to_string();

    // Blockquote
    let blockquote_regex = Regex::new(r"(> ?.*)").unwrap();
    result = blockquote_regex
        .replace_all(&result, |caps: &regex::Captures| {
            format!("\x1b[3m\x1b[34m\x1b[1m{}\x1b[22m\x1b[0m", &caps[1])
        })
        .to_string();

    // Lists (bold magenta number and bullet)
    let list_regex = Regex::new(r"([\d]+\.|-|\*) (.*)").unwrap();
    result = list_regex
        .replace_all(&result, |caps: &regex::Captures| {
            format!("\x1b[35m\x1b[1m{}\x1b[22m\x1b[0m {}", &caps[1], &caps[2])
        })
        .to_string();

    // Block code (black on gray)
    let block_code_regex = Regex::new(r"(?s)```(\w+)?\n(.*?)\n```").unwrap();
    result = block_code_regex
        .replace_all(&result, |caps: &regex::Captures| {
            let lang = if caps.get(1).is_some() { &caps[1] } else { "" };
            format!(
                "\x1b[3m\x1b[1m{}\x1b[22m\x1b[0m\n\x1b[57;107m{}\x1b[0m\n",
                lang, &caps[2]
            )
        })
        .to_string();

    // Inline code (black on gray)
    let inline_code_regex = Regex::new(r"`(.*?)`").unwrap();
    result = inline_code_regex
        .replace_all(&result, |caps: &regex::Captures| {
            format!("\x1b[57;107m{}\x1b[0m", &caps[1])
        })
        .to_string();

    // Headers (cyan bold)
    let header_regex = Regex::new(r"(#{1,6}) (.*?)\n").unwrap();
    result = header_regex
        .replace_all(&result, |caps: &regex::Captures| {
            format!("\x1b[36m\x1b[1m{} {}\x1b[22m\x1b[0m\n", &caps[1], &caps[2])
        })
        .to_string();

    // Headers with a single line of text followed by 2 or more equal signs
    let header_equals_regex = Regex::new(r"(.*?\n={2,}\n)").unwrap();
    result = header_equals_regex
        .replace_all(&result, |caps: &regex::Captures| {
            format!("\x1b[36m\x1b[1m{}\x1b[22m\x1b[0m\n", &caps[1])
        })
        .to_string();

    // Headers with a single line of text followed by 2 or more dashes
    let header_dashes_regex = Regex::new(r"(.*?\n-{2,}\n)").unwrap();
    result = header_dashes_regex
        .replace_all(&result, |caps: &regex::Captures| {
            format!("\x1b[36m\x1b[1m{}\x1b[22m\x1b[0m\n", &caps[1])
        })
        .to_string();

    // Images (blue underlined)
    let image_regex = Regex::new(r"!\[(.*?)\]\((.*?)\)").unwrap();
    result = image_regex
        .replace_all(&result, |caps: &regex::Captures| {
            format!(
                "\x1b[34m{}\x1b[0m (\x1b[34m\x1b[4m{}\x1b[0m)",
                &caps[1], &caps[2]
            )
        })
        .to_string();

    // Links (blue underlined)
    let link_regex = Regex::new(r"!?\[(.*?)\]\((.*?)\)").unwrap();
    result = link_regex
        .replace_all(&result, |caps: &regex::Captures| {
            format!(
                "\x1b[34m{}\x1b[0m (\x1b[34m\x1b[4m{}\x1b[0m)",
                &caps[1], &caps[2]
            )
        })
        .to_string();

    result
}

fn convert_file_to_markdown(file_path: &Path) -> Result<String, String> {
    let md_converter = MarkItDown::new();

    match md_converter.convert(file_path.to_str().unwrap(), None) {
        Some(conversion_result) => Ok(conversion_result.text_content),
        None => Err("Conversion failed or unsupported file type".to_string()),
    }
}

fn is_supported_file_type(file_path: &Path) -> bool {
    if let Some(extension) = file_path.extension() {
        if let Some(ext_str) = extension.to_str() {
            let ext_lower = ext_str.to_lowercase();
            matches!(
                ext_lower.as_str(),
                "pdf" | "xlsx" | "doc" | "docx" | "ppt" | "pptx"
            )
        } else {
            false
        }
    } else {
        false
    }
}

fn clean_whitespace(text: &str) -> String {
    text.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn chunk_markdown_content(content: &str, chunk_size: usize) -> Vec<String> {
    let cleaned_content = clean_whitespace(content);
    let splitter = MarkdownSplitter::new(chunk_size);
    splitter
        .chunks(&cleaned_content)
        .map(|chunk| chunk.to_string())
        .collect()
}

fn generate_dense_embeddings(
    chunks: &[String],
    model: &mut TextEmbedding,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    // Prepare documents with "passage:" prefix for better retrieval performance
    let documents: Vec<String> = chunks
        .iter()
        .map(|chunk| format!("passage: {}", chunk))
        .collect();

    let embeddings = model.embed(documents, None)?;

    Ok(embeddings)
}

fn generate_sparse_embeddings(
    chunks: &[String],
    model: &mut SparseTextEmbedding,
) -> Result<Vec<fastembed::SparseEmbedding>, Box<dyn std::error::Error>> {
    // Prepare documents with "passage:" prefix for better retrieval performance
    let documents: Vec<String> = chunks
        .iter()
        .map(|chunk| format!("passage: {}", chunk))
        .collect();

    let embeddings = model.embed(documents, None)?;

    Ok(embeddings)
}

async fn perform_search(
    vector_store: &qdrant_client::QdrantVectorStore,
    query: &str,
    dense_model: &mut TextEmbedding,
    sparse_model: &mut SparseTextEmbedding,
    reranker: &mut TextRerank,
    ui: &CliUI,
) -> Result<(), Box<dyn std::error::Error>> {
    ui.print_section("Searching");

    let search_spinner = ui.show_loading("Searching vector database...");
    let results = vector_store
        .hybrid_search(query, dense_model, sparse_model, reranker)
        .await?;
    search_spinner.finish_and_clear();

    if results.is_empty() {
        ui.print_warning("No results found.");
        return Ok(());
    }

    ui.print_success(&format!("Found {} results", results.len()));

    let llm_spinner = ui.show_loading("Generating response from LLM...");

    // call LLM with results
    let openai_client = ai::OpenAiClient::new(
        env::var("OPENAI_API_KEY").unwrap(),
        env::var("OPENAI_URL").unwrap(),
        env::var("OPENAI_MODEL").unwrap(),
    );
    let response = openai_client
        .generate_response(query, &results, None)
        .await?;

    llm_spinner.finish_and_clear();

    ui.print_section("AI Response");
    println!("{}", format_markdown(&response).bright_white());

    Ok(())
}

#[tokio::main]
async fn main() {
    let ui = CliUI::new();
    ui.print_header();

    // Load environment variables from .env file if it exists
    dotenv::dotenv().ok();

    let args = Args::parse();

    ui.print_section("Initializing AI Models");
    let init_spinner = ui.show_loading("Loading embedding models...");

    // Initialize Qdrant client
    let vector_store = match qdrant_client::QdrantVectorStore::new("http://localhost:6334").await {
        Ok(store) => {
            init_spinner.finish_and_clear();
            ui.print_success("Connected to Qdrant vector database");
            store
        }
        Err(e) => {
            init_spinner.finish_and_clear();
            ui.print_error(&format!("Failed to connect to Qdrant: {}", e));
            ui.print_error("Make sure Qdrant is running on http://localhost:6334");
            std::process::exit(1);
        }
    };

    let mut dense_model =
        TextEmbedding::try_new(InitOptions::new(EmbeddingModel::BGESmallENV15)).unwrap();
    let mut sparse_model =
        SparseTextEmbedding::try_new(fastembed::SparseInitOptions::new(SparseModel::SPLADEPPV1))
            .unwrap();
    let mut reranker =
        TextRerank::try_new(RerankInitOptions::new(RerankerModel::JINARerankerV1TurboEn)).unwrap();

    init_spinner.finish_and_clear();
    ui.print_success("AI models loaded successfully");

    // Handle search functionality
    if let Some(query) = args.search {
        if let Err(e) = perform_search(
            &vector_store,
            &query,
            &mut dense_model,
            &mut sparse_model,
            &mut reranker,
            &ui,
        )
        .await
        {
            ui.print_error(&format!("Search failed: {}", e));
            std::process::exit(1);
        }
        return;
    }

    let path = Path::new(&args.directory);

    if !path.exists() {
        ui.print_error(&format!("Directory '{}' does not exist", args.directory));
        std::process::exit(1);
    }

    if !path.is_dir() {
        ui.print_error(&format!("'{}' is not a directory", args.directory));
        std::process::exit(1);
    }

    ui.print_section("Directory Discovery");
    ui.print_info(&format!("Crawling directory: {}", args.directory));
    if let Some(since_timestamp) = args.since {
        ui.print_info(&format!(
            "Filtering files modified since: {}",
            since_timestamp
        ));
    }

    // First pass: discover all files
    let mut files_to_process = Vec::new();
    let mut file_infos = Vec::new();

    for entry in WalkDir::new(path) {
        if let Err(e) = entry {
            ui.print_warning(&format!("Error accessing entry: {}", e));
            continue;
        }

        let entry = entry.unwrap();
        let is_file = entry.file_type().is_file();
        let should_include = if let Some(since_timestamp) = args.since {
            match entry.metadata() {
                Ok(metadata) => {
                    match metadata.modified() {
                        Ok(modified_time) => {
                            let unix_timestamp = modified_time
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();
                            unix_timestamp >= since_timestamp
                        }
                        Err(_) => true, // Include if we can't get modification time
                    }
                }
                Err(_) => true, // Include if we can't get metadata
            }
        } else {
            true // Include all files if no filter specified
        };

        if !is_file || !should_include || !is_supported_file_type(entry.path()) {
            continue;
        }

        let metadata = fs::metadata(entry.path()).unwrap();
        let modified_time = metadata
            .modified()
            .unwrap()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let modified_date = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(modified_time);
        let formatted_date = format!("{:?}", modified_date);

        file_infos.push(FileInfo {
            name: entry
                .path()
                .file_name()
                .unwrap_or_default()
                .to_str()
                .unwrap()
                .to_string(),
            size: format!("{} KB", metadata.len() / 1024),
            modified: formatted_date
                .split(' ')
                .next()
                .unwrap_or("Unknown")
                .to_string(),
            status: "Pending".to_string(),
        });

        files_to_process.push(entry);
    }

    ui.print_success(&format!(
        "Found {} files to process",
        files_to_process.len()
    ));
    ui.print_files_table(&file_infos);

    if files_to_process.is_empty() {
        ui.print_warning("No files found to process");
        return;
    }

    ui.print_section("File Processing");
    let file_progress = ui.create_file_progress_bar(files_to_process.len());

    for entry in files_to_process {
        let file_name = entry
            .path()
            .file_name()
            .unwrap_or_default()
            .to_str()
            .unwrap()
            .to_string();
        file_progress.set_message(format!("Processing: {}", file_name));

        let markdown_content = convert_file_to_markdown(entry.path());

        // skip if conversion failed
        if let Err(e) = markdown_content {
            ui.print_warning(&format!(
                "Conversion failed for {}: {}",
                entry.path().display(),
                e
            ));
            file_progress.inc(1);
            continue;
        }

        let markdown_content = markdown_content.unwrap();

        // Store file metadata in Qdrant
        let metadata = fs::metadata(entry.path()).unwrap();
        let modified_time = metadata
            .modified()
            .unwrap()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let content_hash = format!("{:x}", md5::compute(&markdown_content));

        let file_id = match vector_store
            .store_file_metadata(
                entry.path().to_str().unwrap(),
                entry
                    .path()
                    .file_name()
                    .unwrap_or_default()
                    .to_str()
                    .unwrap(),
                metadata.len(),
                modified_time,
                &content_hash,
                Some(&markdown_content),
            )
            .await
        {
            Ok(id) => {
                ui.print_success(&format!("Stored file in Qdrant with ID: {}", id));
                id
            }
            Err(e) => {
                ui.print_error(&format!("Failed to store file in Qdrant: {}", e));
                file_progress.inc(1);
                continue;
            }
        };

        let chunks = chunk_markdown_content(&markdown_content, 1000);

        let embedding_progress = ui.create_embedding_progress_bar(chunks.len());
        embedding_progress.set_message("Generating dense embeddings...");
        let dense_embeddings = generate_dense_embeddings(&chunks, &mut dense_model).unwrap();
        embedding_progress.set_message("Generating sparse embeddings...");
        let sparse_embeddings = generate_sparse_embeddings(&chunks, &mut sparse_model).unwrap();
        embedding_progress.finish_and_clear();

        ui.print_success(&format!(
            "Generated {} dense embeddings",
            dense_embeddings.len()
        ));
        ui.print_success(&format!(
            "Generated {} sparse embeddings",
            sparse_embeddings.len()
        ));

        if dense_embeddings.len() != sparse_embeddings.len() {
            ui.print_error("Dense and sparse embeddings have different lengths");
            file_progress.inc(1);
            continue;
        }

        let store_progress = ui.create_spinner("Storing embeddings in vector database...");
        for (i, dense_embedding) in dense_embeddings.iter().enumerate() {
            let sparse_embedding = &sparse_embeddings[i];
            vector_store
                .store_embeddings(
                    file_id.as_str(),
                    &chunks[i],
                    i as i32,
                    dense_embedding,
                    sparse_embedding,
                )
                .await
                .unwrap();
        }
        store_progress.finish_and_clear();

        file_progress.inc(1);
    }

    file_progress.finish_and_clear();
    ui.print_completion("All files processed successfully!");
}
