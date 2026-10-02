# Pointing ALEXIAI at a local model

ALEXIAI speaks to any server that exposes the OpenAI-compatible surface on
loopback. The reference shape is **osarous** — the Apple Silicon MLX sidecar
(`llm-osarous` in the DeepSeek harness) at `http://127.0.0.1:1337`.

## the shape the adapter expects

```
GET  {base}/v1/models                     → { "data": [ { "id", "name", "context_window" } ] }
POST {base}/v1/chat/completions           → OpenAI body, SSE stream when stream:true
GET  {base}/health                        → 200 when up
```

## run

```bash
# with the default endpoint
cargo run --release -p alexiai -- serve

# with your own local port
cargo run --release -p alexiai -- serve --endpoint http://127.0.0.1:8080

# sanity
cargo run --release -p alexiai -- doctor --endpoint http://127.0.0.1:8080
cargo run --release -p alexiai -- models --endpoint http://127.0.0.1:8080
cargo run --release -p alexiai -- chat "hello" --endpoint http://127.0.0.1:8080

# through the Go glue
go run ./golue run --endpoint http://127.0.0.1:8080
```

A remote `--endpoint` is refused at construction — `sovereignty violation` —
with a one-line explanation. That is the product working as designed.
