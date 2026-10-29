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
    let local_state = tempfile::tempdir().unwrap();
    std::fs::write(local_state.path().join("metadata.json"), r#"{"storage_backend":"postgres","storage_repo":"alpha","model_short_name":"test-model","dimensions":3}"#).unwrap();
    assert!(!local_state.path().join("data.mdb").exists());
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
    let mut backend = codesearch::vectordb::ChunkStore::Postgres {
        store: store.clone(),
        repo: "alpha".into(),
    };
    assert_eq!(
        backend
            .get_chunk_as_result(replacement[0])
            .await
            .unwrap()
            .unwrap()
            .content,
        "replacement"
    );
    assert_eq!(
        backend.chunks_for_file("src/main.rs").await.unwrap().len(),
        1
    );
    assert_eq!(
        backend.get_chunks_by_file().await.unwrap()["src/main.rs"],
        replacement
    );
    assert_eq!(backend.index_health().await.unwrap(), (1, true));
    backend.build_index().await.unwrap();
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
    // Missing local publication state must not erase a committed remote file,
    // even when embedding cannot proceed.
    let recovery = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(recovery.path().join("src")).unwrap();
    std::fs::write(recovery.path().join("src/main.rs"), "fn replacement() {}\n").unwrap();
    let recovery_db = recovery.path().join(".codesearch.db");
    let shared = codesearch::index::SharedStores::new(&recovery_db, 3).unwrap();
    *shared.vector_store.write().await = codesearch::vectordb::ChunkStore::Postgres {
        store: store.clone(),
        repo: "alpha".into(),
    };
    std::fs::write(
        recovery_db.join("metadata.json"),
        r#"{"model_short_name":"test-model","dimensions":3}"#,
    )
    .unwrap();
    let result = codesearch::index::IndexManager::perform_incremental_refresh_with_stores(
        recovery.path(),
        &recovery_db,
        &shared,
        &tokio_util::sync::CancellationToken::new(),
        None,
        None,
    )
    .await;
    assert!(
        result.is_err(),
        "unavailable embedding model must fail refresh"
    );
    assert_eq!(
        store
            .get_chunk("alpha", replacement[0])
            .await
            .unwrap()
            .unwrap()
            .content,
        "replacement"
    );

    // Sparse path scopes must fill the requested limit despite closer vectors elsewhere.
    let distractors: Vec<_> = (0..300)
        .map(|_| chunk("bulk.rs", "distractor", [1.0, 0.0, 0.0]))
        .collect();
    store
        .replace_file("noise", "bulk.rs", &distractors)
        .await
        .unwrap();
    let scoped: Vec<_> = (0..12)
        .map(|_| chunk("selected.rs", "scoped", [0.1, 0.9, 0.0]))
        .collect();
    store
        .replace_file("scope", "selected.rs", &scoped)
        .await
        .unwrap();
    let hits = store
        .search(
            &["noise".into(), "scope".into()],
            &[1.0, 0.0, 0.0],
            Some("selected"),
            10,
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 10);
    assert!(hits
        .iter()
        .all(|hit| hit.origin.as_deref() == Some("scope")));
    store.delete_file("alpha", "src/main.rs").await.unwrap();
    assert_eq!(store.stats("alpha").await.unwrap().total_chunks, 0);
    store.delete_repo("beta").await.unwrap();
    assert_eq!(store.stats("beta").await.unwrap().total_chunks, 0);

    let pool = sqlx::PgPool::connect(&url).await.unwrap();
    sqlx::query("CREATE DATABASE cli_fixture")
        .execute(&pool)
        .await
        .unwrap();
    let cli_url = url.replace("/postgres?", "/cli_fixture?");
    let cli_store = PostgresStore::connect(&cli_url, "minilm-l6-q", 384)
        .await
        .unwrap();
    let project = tempfile::tempdir().unwrap();
    let db = project.path().join(".codesearch.db");
    std::fs::create_dir_all(&db).unwrap();
    std::fs::write(db.join("metadata.json"), r#"{"schema_version":1,"storage_backend":"postgres","storage_repo":"cli","model_short_name":"minilm-l6-q","dimensions":384,"partial":false}"#).unwrap();
    let ids = cli_store
        .replace_file(
            "cli",
            "src/rule.xml",
            &[EmbeddedChunk::new(
                Chunk::new(
                    "<ref:classification search=\"gadget\" />".into(),
                    1,
                    1,
                    ChunkKind::Block,
                    "src/rule.xml".into(),
                ),
                vec![0.5; 384],
            )],
        )
        .await
        .unwrap();
    let mut fts = codesearch::fts::FtsStore::new_with_writer(&db).unwrap();
    fts.add_chunk(
        ids[0],
        "<ref:classification search=\"gadget\" />",
        "src/rule.xml",
        None,
        "Block",
    )
    .unwrap();
    fts.commit().unwrap();
    drop(fts);
    for args in [
        vec!["gadget", "--literal"],
        vec![r"ref:classification\s+search=", "--regex"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_codesearch"))
            .arg("search")
            .args(args)
            .arg("--json")
            .arg("--path")
            .arg(project.path())
            .env("CODESEARCH_DATABASE_URL", &cli_url)
            .env("CODESEARCH_STORAGE", "postgres")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "CLI search failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            output.stderr.is_empty(),
            "unexpected CLI diagnostics: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let response: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(response["results"][0]["path"], "src/rule.xml");
    }
    assert!(!db.join("data.mdb").exists());
}
