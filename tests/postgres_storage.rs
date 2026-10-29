#![cfg(feature = "pgvector")]

use codesearch::vectordb::PostgresStore;
use codesearch::{Chunk, ChunkKind, EmbeddedChunk};
use std::process::Command;

struct Database(String);

impl Drop for Database {
    fn drop(&mut self) {
        let output = Command::new("docker").args(["rm", "-f", &self.0]).output();
        if !matches!(output, Ok(ref result) if result.status.success()) {
            eprintln!(
                "Failed to remove disposable pgvector test container {}",
                self.0
            );
        }
    }
}

impl Database {
    async fn start() -> (Self, String) {
        let name = format!("codesearch-pgvector-{}", uuid::Uuid::new_v4());
        let output = Command::new("docker")
            .args([
                "run",
                "--detach",
                "--rm",
                "--name",
                &name,
                "-e",
                "POSTGRES_HOST_AUTH_METHOD=trust",
                "-p",
                "127.0.0.1::5432",
                "pgvector/pgvector:0.8.0-pg17",
            ])
            .output()
            .expect("Docker is required for pgvector integration tests");
        assert!(
            output.status.success(),
            "container startup failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let database = Self(name);
        let output = Command::new("docker")
            .args(["port", &database.0, "5432/tcp"])
            .output()
            .expect("inspect test container port");
        assert!(
            output.status.success(),
            "container port lookup failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let address = String::from_utf8(output.stdout).expect("container address");
        let url = format!(
            "postgres://postgres@{}/postgres?sslmode=disable",
            address.trim()
        );
        for _ in 0..120 {
            let output = Command::new("docker")
                .args([
                    "exec",
                    &database.0,
                    "pg_isready",
                    "-h",
                    "127.0.0.1",
                    "-U",
                    "postgres",
                ])
                .output()
                .expect("probe test database");
            if output.status.success() {
                return (database, url);
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        panic!("pgvector test database did not become ready");
    }
}

fn chunk(path: &str, content: &str, embedding: [f32; 3]) -> EmbeddedChunk {
    EmbeddedChunk::new(
        Chunk::new(
            content.to_owned(),
            1,
            2,
            ChunkKind::Function,
            path.to_owned(),
        ),
        embedding.to_vec(),
    )
}

#[tokio::test]
async fn postgres_file_lifecycle_and_filtered_search() {
    let (_database, url) = Database::start().await;
    let store = PostgresStore::connect(&url, "test-model", 3).await.unwrap();
    // Opening repeatedly exercises idempotent schema initialization.
    let reopened = PostgresStore::connect(&url, "test-model", 3).await.unwrap();
    let ids = store
        .replace_file(
            "alpha",
            "src/main.rs",
            &[chunk("src/main.rs", "first", [1.0, 0.0, 0.0])],
        )
        .await
        .unwrap();
    store
        .replace_file(
            "beta",
            "lib.rs",
            &[chunk("lib.rs", "other repo", [0.0, 1.0, 0.0])],
        )
        .await
        .unwrap();
    assert_eq!(
        reopened
            .get_chunk("alpha", ids[0])
            .await
            .unwrap()
            .unwrap()
            .content,
        "first"
    );
    assert_eq!(
        store.get_embedding("alpha", ids[0]).await.unwrap().unwrap(),
        vec![1.0, 0.0, 0.0]
    );
    let hits = store
        .search(&["alpha".into(), "beta".into()], &[1.0, 0.0, 0.0], None, 2)
        .await
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].origin.as_deref(), Some("alpha"));
    let hits = store
        .search(&["alpha".into()], &[1.0, 0.0, 0.0], Some("src/"), 5)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(store
        .search(&["alpha".into()], &[1.0, 0.0, 0.0], Some("lib"), 5)
        .await
        .unwrap()
        .is_empty());
    let replacement = store
        .replace_file(
            "alpha",
            "src/main.rs",
            &[chunk("src/main.rs", "replacement", [0.0, 0.0, 1.0])],
        )
        .await
        .unwrap();
    assert_ne!(ids, replacement);
    assert!(store.get_chunk("alpha", ids[0]).await.unwrap().is_none());
    assert_eq!(store.stats("alpha").await.unwrap().total_chunks, 1);
    // Invalid replacement must preserve the complete previous file.
    let invalid = EmbeddedChunk::new(
        Chunk::new(
            "invalid".into(),
            1,
            1,
            ChunkKind::Function,
            "src/main.rs".into(),
        ),
        vec![1.0, 2.0],
    );
    assert!(store
        .replace_file("alpha", "src/main.rs", &[invalid])
        .await
        .is_err());
    assert_eq!(
        store
            .get_chunk("alpha", replacement[0])
            .await
            .unwrap()
            .unwrap()
            .content,
        "replacement"
    );
    assert!(PostgresStore::connect(&url, "test-model", 4)
        .await
        .err()
        .unwrap()
        .to_string()
        .contains("mismatch"));
    assert!(PostgresStore::connect(&url, "different-model", 3)
        .await
        .err()
        .unwrap()
        .to_string()
        .contains("mismatch"));
    store.delete_file("alpha", "src/main.rs").await.unwrap();
    assert_eq!(store.stats("alpha").await.unwrap().total_chunks, 0);
    store.delete_repo("beta").await.unwrap();
    assert_eq!(store.stats("beta").await.unwrap().total_chunks, 0);
}
