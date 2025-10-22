use qdrant_client::{
    qdrant::{
        CreateCollectionBuilder, Distance, Fusion, NamedVectors, PointStruct, PrefetchQueryBuilder,
        Query, QueryPointsBuilder, SparseVectorParamsBuilder, SparseVectorsConfigBuilder,
        UpsertPointsBuilder, Vector, VectorParamsBuilder, VectorsConfigBuilder,
    },
    Payload, Qdrant,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Instant;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub file_path: String,
    pub file_name: String,
    pub file_size: u64,
    pub modified_time: u64,
    pub content_hash: String,
    pub markdown_content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub file_id: String,
    pub chunk_index: i32,
    pub chunk_content: String,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub file_path: String,
    pub file_name: String,
    pub chunk_content: String,
    pub chunk_index: i32,
    pub similarity_score: f64,
}

pub struct QdrantVectorStore {
    client: Qdrant,
    collection_name: String,
    files_collection_name: String,
}

const SPARSE_NAME: &str = "text-sparse";
const DENSE_NAME: &str = "text-dense";

impl QdrantVectorStore {
    pub async fn new(url: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let client = Qdrant::from_url(url).build()?;

        let store = Self {
            client,
            collection_name: "file_embeddings".to_string(),
            files_collection_name: "files".to_string(),
        };

        // Initialize collections
        store.init_collections().await?;

        Ok(store)
    }

    async fn init_collections(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Create files collection (for metadata storage)
        let files_response = self
            .client
            .create_collection(
                CreateCollectionBuilder::new(&self.files_collection_name)
                    .vectors_config(VectorParamsBuilder::new(1, Distance::Cosine)),
            )
            .await;

        match files_response {
            Ok(_) => println!("Files collection created successfully"),
            Err(e) => println!("Files collection creation result: {:?}", e),
        }

        let mut vector_config = VectorsConfigBuilder::default();
        vector_config
            .add_named_vector_params(DENSE_NAME, VectorParamsBuilder::new(384, Distance::Cosine));

        let mut sparse_vector_config = SparseVectorsConfigBuilder::default();

        sparse_vector_config
            .add_named_vector_params(SPARSE_NAME, SparseVectorParamsBuilder::default());

        // Create file embeddings collection
        let dense_response = self
            .client
            .create_collection(
                CreateCollectionBuilder::new(&self.collection_name)
                    .vectors_config(vector_config)
                    .sparse_vectors_config(sparse_vector_config),
            )
            .await;

        match dense_response {
            Ok(_) => println!("Dense collection created successfully"),
            Err(e) => println!("Dense collection creation result: {:?}", e),
        }

        Ok(())
    }

    pub async fn store_file_metadata(
        &self,
        file_path: &str,
        file_name: &str,
        file_size: u64,
        modified_time: u64,
        content_hash: &str,
        markdown_content: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let file_id = format!("{:x}", md5::compute(file_path));

        let metadata = FileMetadata {
            file_path: file_path.to_string(),
            file_name: file_name.to_string(),
            file_size,
            modified_time,
            content_hash: content_hash.to_string(),
            markdown_content: markdown_content.map(|s| s.to_string()),
        };

        let point = PointStruct::new(
            file_id.clone(),
            vec![0.0], // Dummy vector for metadata collection
            Payload::try_from(json!({
                "file_path": metadata.file_path,
                "file_name": metadata.file_name,
                "file_size": metadata.file_size as f64,
                "modified_time": metadata.modified_time as f64,
                "content_hash": metadata.content_hash,
                "markdown_content": metadata.markdown_content.unwrap_or_default(),
            }))
            .unwrap(),
        );

        self.client
            .upsert_points(UpsertPointsBuilder::new(
                &self.files_collection_name,
                vec![point],
            ))
            .await?;

        Ok(file_id)
    }

    pub async fn store_embeddings(
        &self,
        file_id: &str,
        chunk: &String,
        chunk_index: i32,
        dense_embedding: &Vec<f32>,
        sparse_embedding: &fastembed::SparseEmbedding,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let embeddings_response = self.client.upsert_points(
            UpsertPointsBuilder::new(
                &self.collection_name,
                vec![PointStruct::new(
                    Uuid::new_v4().to_string(),
                    NamedVectors::default()
                        .add_vector(DENSE_NAME, Vector::new_dense(dense_embedding.clone()))
                        .add_vector(
                            SPARSE_NAME,
                            Vector::new_sparse(
                                sparse_embedding
                                    .indices
                                    .clone()
                                    .into_iter()
                                    .map(|i| i as u32)
                                    .collect::<Vec<u32>>(),
                                sparse_embedding.values.clone(),
                            ),
                        ),
                    Payload::try_from(json!({
                        "file_id": file_id.to_string(),
                        "chunk_index": chunk_index as f64,
                        "chunk_content": chunk,
                    }))
                    .unwrap(),
                )],
            )
            .wait(true),
        );

        match embeddings_response.await {
            Ok(_) => {}
            Err(e) => println!("Embeddings storage error result: {:?}", e),
        };

        Ok(())
    }

    pub async fn hybrid_search(
        &self,
        query: &str,
        dense_model: &mut fastembed::TextEmbedding,
        sparse_model: &mut fastembed::SparseTextEmbedding,
        reranker: &mut fastembed::TextRerank,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let overall_start = Instant::now();

        // Generate sparse embeddings
        let sparse_start = Instant::now();
        let sparse_query_embeddings = sparse_model.embed(vec![query.to_string()], None)?;
        let sparse_duration = sparse_start.elapsed();
        println!("Sparse embedding generation: {:?}", sparse_duration);

        // Generate dense embeddings
        let dense_start = Instant::now();
        let dense_query_embeddings = dense_model.embed(vec![query.to_string()], None)?;
        let dense_duration = dense_start.elapsed();
        println!("Dense embedding generation: {:?}", dense_duration);

        let sparse_query_embedding = sparse_query_embeddings.first().unwrap();
        let query_sparse = qdrant_client::qdrant::VectorInput::new_sparse(
            sparse_query_embedding
                .indices
                .clone()
                .into_iter()
                .map(|i| i as u32)
                .collect::<Vec<u32>>(),
            sparse_query_embedding.values.clone(),
        );
        let dense_query_embedding = dense_query_embeddings.first().unwrap();
        let query_dense =
            qdrant_client::qdrant::VectorInput::new_dense(dense_query_embedding.clone());

        // Vector search query execution
        let search_start = Instant::now();
        // 50 total results => 25 results from each embedding type, sorted by score
        let vector_results = self
            .client
            .query(
                QueryPointsBuilder::new(&self.collection_name)
                    .add_prefetch(
                        PrefetchQueryBuilder::default()
                            .query(Query::new_nearest(query_sparse))
                            .using(SPARSE_NAME) // sparse embedding
                            .limit(25 as u64),
                    )
                    .add_prefetch(
                        PrefetchQueryBuilder::default()
                            .query(Query::new_nearest(query_dense))
                            .using(DENSE_NAME) // dense embedding
                            .limit(25 as u64),
                    )
                    .query(Query::new_fusion(Fusion::Rrf))
                    .limit(50 as u64)
                    .with_payload(true),
            )
            .await?;
        let search_duration = search_start.elapsed();
        println!("Vector search query execution: {:?}", search_duration);

        let documents = vector_results
            .result
            .iter()
            .filter_map(|result| {
                result
                    .payload
                    .get("chunk_content")
                    .and_then(|v| v.as_str().map(|s| s.as_str()))
            })
            .collect::<Vec<&str>>();

        println!("Documents found: {}", documents.len());

        // Reranking step
        let rerank_start = Instant::now();
        println!("Reranking documents...");
        // Re-rank the results using the reranker and take the top 10
        let reranked_results = reranker.rerank(query, documents, true, None)?;
        let rerank_duration = rerank_start.elapsed();
        println!("Reranking: {:?}", rerank_duration);

        let final_results = reranked_results.iter().take(10).collect::<Vec<_>>();

        let overall_duration = overall_start.elapsed();
        println!("Total hybrid search time: {:?}", overall_duration);

        Ok(final_results
            .into_iter()
            .map(|result| SearchResult {
                file_path: "".to_string(),
                file_name: "".to_string(),
                chunk_content: result.document.clone().unwrap_or_default(),
                chunk_index: 0,
                similarity_score: result.score as f64,
            })
            .collect::<Vec<_>>())
    }
}
