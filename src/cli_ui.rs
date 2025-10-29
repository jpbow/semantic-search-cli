use colored::*;
use console::Term;
use dialoguer::{Confirm, Input, Select};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::time::Duration;
use tabled::{
    settings::{Alignment, Style},
    Table, Tabled,
};

/// Modern CLI UI utilities for enhanced user experience
pub struct CliUI {
    multi_progress: MultiProgress,
    term: Term,
}

impl CliUI {
    pub fn new() -> Self {
        Self {
            multi_progress: MultiProgress::new(),
            term: Term::stdout(),
        }
    }

    /// Print a beautiful header with the app name
    pub fn print_header(&self) {
        println!();
        println!("{}", "=".repeat(80).bright_blue());
        println!(
            "{}",
            "🚀 FILE CRAWLER & SEMANTIC SEARCH".bright_cyan().bold()
        );
        println!("{}", "=".repeat(80).bright_blue());
        println!();
    }

    /// Print a section header with styling
    pub fn print_section(&self, title: &str) {
        println!();
        println!("{}", format!("📁 {}", title).bright_yellow().bold());
        println!("{}", "─".repeat(title.len() + 4).bright_yellow());
    }

    /// Print a success message with checkmark
    pub fn print_success(&self, message: &str) {
        println!("{} {}", "✅".green(), message.green());
    }

    /// Print an error message with X mark
    pub fn print_error(&self, message: &str) {
        println!("{} {}", "❌".red(), message.red());
    }

    /// Print a warning message with warning sign
    pub fn print_warning(&self, message: &str) {
        println!("{} {}", "⚠️".yellow(), message.yellow());
    }

    /// Print an info message with info icon
    pub fn print_info(&self, message: &str) {
        println!("{} {}", "ℹ️".blue(), message.blue());
    }

    /// Create a progress bar for file processing
    pub fn create_file_progress_bar(&self, total_files: usize) -> ProgressBar {
        let pb = self
            .multi_progress
            .add(ProgressBar::new(total_files as u64));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message("Processing files...");
        pb
    }

    /// Create a progress bar for embedding generation
    pub fn create_embedding_progress_bar(&self, total_chunks: usize) -> ProgressBar {
        let pb = self
            .multi_progress
            .add(ProgressBar::new(total_chunks as u64));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.magenta} [{elapsed_precise}] [{bar:40.magenta/blue}] {pos}/{len} chunks {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message("Generating embeddings...");
        pb
    }

    /// Create a spinner for async operations
    pub fn create_spinner(&self, message: &str) -> ProgressBar {
        let pb = self.multi_progress.add(ProgressBar::new_spinner());
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        pb.set_message(message.to_string());
        pb.enable_steady_tick(Duration::from_millis(100));
        pb
    }

    /// Print a table of files being processed
    pub fn print_files_table(&self, files: &[FileInfo]) {
        if files.is_empty() {
            self.print_warning("No files found to process");
            return;
        }

        let table_str = Table::new(files)
            .with(Style::modern())
            .with(Alignment::left())
            .to_string();

        println!("{}", table_str);
    }

    /// Print search results in a beautiful format
    pub fn print_search_results(&self, query: &str, results: &[SearchResult]) {
        println!();
        println!(
            "{}",
            format!("🔍 Search Results for: '{}'", query)
                .bright_cyan()
                .bold()
        );
        println!("{}", "=".repeat(60).bright_cyan());

        if results.is_empty() {
            self.print_warning("No results found");
            return;
        }

        for (i, result) in results.iter().enumerate() {
            println!();
            println!(
                "{} {}",
                format!("{}.", i + 1).bright_yellow().bold(),
                result.file_name.bright_white().bold()
            );
            println!("{} {}", "📄 File:".blue(), result.file_path.blue());
            println!(
                "{} {}",
                "📊 Score:".green(),
                format!("{:.3}", result.score).green()
            );
            println!("{}", "─".repeat(50).dimmed());
            println!("{}", result.content.trim().white());
            println!();
        }
    }

    /// Ask for user confirmation
    pub fn ask_confirmation(&self, prompt: &str) -> bool {
        Confirm::new()
            .with_prompt(prompt)
            .default(true)
            .interact()
            .unwrap_or(false)
    }

    /// Ask for user input
    pub fn ask_input(&self, prompt: &str) -> String {
        Input::<String>::new()
            .with_prompt(prompt)
            .interact_text()
            .unwrap_or_default()
    }

    /// Show a selection menu
    pub fn show_selection(&self, prompt: &str, items: &[String]) -> Option<usize> {
        Select::new()
            .with_prompt(prompt)
            .items(items)
            .interact_opt()
            .ok()
            .flatten()
    }

    /// Print a loading message with spinner
    pub fn show_loading(&self, message: &str) -> ProgressBar {
        let pb = self.create_spinner(message);
        pb
    }

    /// Print a completion message with celebration
    pub fn print_completion(&self, message: &str) {
        println!();
        println!("🎉 {}", message.bright_green().bold());
        println!();
    }

    /// Clear the screen
    pub fn clear_screen(&self) {
        self.term.clear_screen().ok();
    }

    /// Print a separator line
    pub fn print_separator(&self) {
        println!("{}", "─".repeat(80).dimmed());
    }
}

/// File information for table display
#[derive(Tabled)]
pub struct FileInfo {
    #[tabled(rename = "File Name")]
    pub name: String,
    #[tabled(rename = "Size")]
    pub size: String,
    #[tabled(rename = "Modified")]
    pub modified: String,
    #[tabled(rename = "Status")]
    pub status: String,
}

/// Search result information
pub struct SearchResult {
    pub file_name: String,
    pub file_path: String,
    pub content: String,
    pub score: f32,
}

impl Default for CliUI {
    fn default() -> Self {
        Self::new()
    }
}
