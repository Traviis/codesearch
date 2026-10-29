use anyhow::{bail, Result};
use std::path::Path;

use super::{ChunkMetadata, SearchResult, StoreStats, VectorStore};
use crate::embed::EmbeddedChunk;

/// Storage for a repository's vectors and complete indexed chunk payloads.
pub enum ChunkStore {
    Local(VectorStore),
    #[cfg(feature = "pgvector")]
    Postgres {
        store: super::PostgresStore,
        repo: String,
    },
}

impl ChunkStore {
    pub async fn open(
        db_path: &Path,
        repo: &str,
        model: &str,
        dimensions: usize,
        readonly: bool,
    ) -> Result<Self> {
        match std::env::var("CODESEARCH_STORAGE")
            .as_deref()
            .unwrap_or("local")
        {
            "local" => Ok(Self::Local(if readonly {
                VectorStore::open_readonly(db_path, dimensions)?
            } else {
                VectorStore::new(db_path, dimensions)?
            })),
            "postgres" => {
                #[cfg(feature = "pgvector")]
                {
                    let url = std::env::var("CODESEARCH_DATABASE_URL").map_err(|_| {
                        anyhow::anyhow!("Postgres storage requires CODESEARCH_DATABASE_URL")
                    })?;
                    let store = super::PostgresStore::connect(&url, model, dimensions).await?;
                    Ok(Self::Postgres {
                        store,
                        repo: repo.to_owned(),
                    })
                }
                #[cfg(not(feature = "pgvector"))]
                {
                    let _ = (repo, model);
                    bail!("Postgres storage requires a build with --features pgvector");
                }
            }
            _ => bail!("CODESEARCH_STORAGE must be 'local' or 'postgres'"),
        }
    }

    pub async fn open_for_path(
        db_path: &Path,
        project_path: &Path,
        model: &str,
        dimensions: usize,
        readonly: bool,
    ) -> Result<Self> {
        if std::env::var("CODESEARCH_STORAGE").as_deref().unwrap_or("local") == "local" {
            return Self::open(db_path, "", model, dimensions, readonly).await;
        }
        let config = crate::db_discovery::repos::ReposConfig::load()?;
        let normalized = crate::cache::normalize_path(project_path);
        let repo = config
            .repos
            .iter()
            .find_map(|(alias, entry)| {
                (crate::cache::normalize_path(entry) == normalized).then(|| alias.clone())
            })
            .unwrap_or_else(|| normalized.clone());
        Self::open(db_path, &repo, model, dimensions, readonly).await
    }

    pub fn backend(&self) -> &'static str {
        match self {
            Self::Local(_) => "local",
            #[cfg(feature = "pgvector")]
            Self::Postgres { .. } => "postgres",
        }
    }

    pub async fn search(
        &self,
        embedding: &[f32],
        prefix: Option<&str>,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        match self {
            Self::Local(store) => {
                let mut results = store.search(embedding, limit)?;
                if let Some(prefix) = prefix {
                    results.retain(|r| r.path.starts_with(prefix));
                }
                Ok(results)
            }
            #[cfg(feature = "pgvector")]
            Self::Postgres { store, repo } => {
                store
                    .search(std::slice::from_ref(repo), embedding, prefix, limit)
                    .await
            }
        }
    }

    pub async fn get_chunk(&self, id: u32) -> Result<Option<ChunkMetadata>> {
        match self {
            Self::Local(store) => store.get_chunk(id),
            #[cfg(feature = "pgvector")]
            Self::Postgres { store, repo } => store.get_chunk(repo, id).await,
        }
    }

    pub async fn get_embedding(&self, id: u32) -> Result<Option<Vec<f32>>> {
        match self {
            Self::Local(store) => store.get_embedding(id),
            #[cfg(feature = "pgvector")]
            Self::Postgres { store, repo } => store.get_embedding(repo, id).await,
        }
    }

    pub async fn get_chunk_as_result(&self, id: u32) -> Result<Option<SearchResult>> {
        Ok(self
            .get_chunk(id)
            .await?
            .map(|chunk| chunk.into_result(id, 1.0, None)))
    }

    pub async fn replace_file(
        &mut self,
        path: &str,
        chunks: Vec<EmbeddedChunk>,
    ) -> Result<Vec<u32>> {
        match self {
            Self::Local(store) => {
                let ids = store.get_chunks_by_file()?.remove(path).unwrap_or_default();
                store.delete_chunks(&ids)?;
                store.insert_chunks_with_ids(chunks)
            }
            #[cfg(feature = "pgvector")]
            Self::Postgres { store, repo } => store.replace_file(repo, path, &chunks).await,
        }
    }

    pub async fn delete_file(&mut self, path: &str) -> Result<()> {
        match self {
            Self::Local(store) => {
                let ids = store.get_chunks_by_file()?.remove(path).unwrap_or_default();
                store.delete_chunks(&ids)?;
                Ok(())
            }
            #[cfg(feature = "pgvector")]
            Self::Postgres { store, repo } => store.delete_file(repo, path).await,
        }
    }

    pub async fn clear(&mut self) -> Result<()> {
        match self {
            Self::Local(store) => store.clear(),
            #[cfg(feature = "pgvector")]
            Self::Postgres { store, repo } => store.delete_repo(repo).await,
        }
    }

    pub async fn stats(&self) -> Result<StoreStats> {
        match self {
            Self::Local(store) => store.stats(),
            #[cfg(feature = "pgvector")]
            Self::Postgres { store, repo } => store.stats(repo).await,
        }
    }

    pub async fn build_index(&mut self) -> Result<()> {
        match self {
            Self::Local(store) => store.build_index(),
            #[cfg(feature = "pgvector")]
            Self::Postgres { .. } => Ok(()),
        }
    }

    pub async fn iter_all_chunks(&self) -> Result<Vec<(u32, ChunkMetadata)>> {
        match self {
            Self::Local(store) => store.iter_all_chunks(),
            #[cfg(feature = "pgvector")]
            Self::Postgres { store, repo } => store.file_chunks(repo, None).await,
        }
    }
}
