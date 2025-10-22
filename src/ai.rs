use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<i32>,
}

#[derive(Debug, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponse {
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

pub struct OpenAiClient {
    client: Client,
    api_key: String,
    url: String,
    model: String,
}

impl OpenAiClient {
    pub fn new(api_key: String, url: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            url,
            model,
        }
    }

    pub async fn generate_response(
        &self,
        query: &str,
        search_results: &[crate::qdrant_client::SearchResult],
        system_message: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Prepare context from search results
        let context = search_results
            .iter()
            .enumerate()
            .map(|(i, result)| {
                format!(
                    "[Source {}] File: {} (Score: {:.4})\nContent: {}\n",
                    i + 1,
                    result.file_name,
                    result.similarity_score,
                    result.chunk_content
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let system_message = system_message.unwrap_or("You are a helpful assistant that analyzes search results from a document database and provides comprehensive answers based on the information found.");

        let user_content = format!(
            "Based on the following search results from a document database, please provide a comprehensive answer to the user's query.\n\nUser Query: {}\n\nSearch Results:\n{}\n\nPlease provide a detailed answer based on the information found in the search results. If the search results don't contain enough information to fully answer the query, please indicate what additional information might be needed.",
            query, context
        );

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: system_message.to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: user_content,
                },
            ],
            temperature: Some(0.7),
            max_tokens: Some(4096),
        };

        let response = self
            .client
            .post(&self.url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("API request failed: {}", error_text).into());
        }

        let chat_response: ChatCompletionResponse = response.json().await?;

        if let Some(choice) = chat_response.choices.first() {
            if let Some(content) = &choice.message.content {
                Ok(content.clone())
            } else {
                Err(format!("Response was truncated (finish_reason: {}). Try increasing max_tokens or reducing the input size.", choice.finish_reason).into())
            }
        } else {
            Err("No choices in response".into())
        }
    }
}
