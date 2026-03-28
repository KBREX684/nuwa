# Security Policy | 安全策略

**Nuwa v0.2.0** -- AI Agent Trainer Framework

## Reporting a Vulnerability | 漏洞报告

We take security vulnerabilities seriously. If you discover a security issue in
Nuwa, please report it responsibly so we can address it promptly.

我们非常重视安全漏洞。如果您在 Nuwa 中发现安全问题，请负责任地披露。

### How to Report | 报告方式

- **GitHub Issues (Private)**: Use the
  [GitHub Security Advisory](https://github.com/KBREX684/nuwa/security/advisories/new)
  to report vulnerabilities privately.
- **Email**: Send details to the project maintainers via the contact information
  in the repository. Please include:
  - A description of the vulnerability
  - Steps to reproduce
  - Affected versions
  - Any proposed fix or mitigation

**Please do not** file public GitHub Issues for security vulnerabilities.

**请不要**在公开的 GitHub Issue 中报告安全漏洞。

### Response Timeline | 响应时间

| Stage | Expected Time |
|-------|--------------|
| Acknowledgment | Within 48 hours |
| Initial Assessment | Within 5 business days |
| Fix / Patch | Depends on severity, typically within 14 days |
| Disclosure | After fix is released |

---

## Supported Versions | 支持的版本

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes |
| < 0.2.0 | No |

Only the latest patch release within the 0.2.x line receives security updates.

只有 0.2.x 系列中的最新补丁版本会获得安全更新。

---

## Security Features | 已实施的安全措施

Nuwa includes several built-in security mechanisms for the web dashboard API:

Nuwa 的 Web 控制台 API 包含多项内置安全机制：

### API Key Authentication | API 密钥认证

Set the `NUWA_API_KEY` environment variable to enable authentication on all
`/api/*` endpoints. When enabled, requests must include an `X-API-Key` header
or `Authorization: Bearer <key>` header matching the configured value.

```bash
export NUWA_API_KEY="your-secret-key-here"
nuwa web
```

When `NUWA_API_KEY` is not set, authentication is disabled (suitable for local
development only).

### CORS Validation | CORS 验证

Cross-Origin Resource Sharing is configured via the `NUWA_CORS_ORIGINS`
environment variable:

```bash
# Allow specific origins (recommended)
export NUWA_CORS_ORIGINS="https://dashboard.example.com,https://admin.example.com"

# Allow all origins (local development only)
export NUWA_CORS_ORIGINS="*"
```

Default (when unset): `["http://localhost:8080", "http://127.0.0.1:8080"]`.

### Rate Limiting | 速率限制

An in-memory, per-client-IP rate limiter is active on all `/api/*` endpoints:
- **Window**: 60 seconds
- **Max requests**: 120 per window
- **Response on exceed**: HTTP 429 with error message

### Path Traversal Protection | 路径遍历防护

The SPA static file server resolves all file paths and verifies they fall under
the expected static directory. Requests attempting to traverse outside the
static root (`../` attacks) are rejected with HTTP 404.

### Input Sanitization | 输入净化

- All request bodies are validated through **Pydantic v2** models with strict
  type checking.
- Secrets (API keys) are stored as `SecretStr` to prevent accidental logging or
  serialization.
- Error responses to clients are generic and never expose internal exception
  text or stack traces.
- Branch name validation uses a strict allowlist regex (`^[A-Za-z0-9._/\-]+$`).

### Circuit Breaker | 熔断器

The LLM backend implements a circuit breaker pattern:
- **Threshold**: 5 consecutive failures
- **Recovery timeout**: 60 seconds
- **Behavior**: Blocks LLM calls during the open state, preventing cascading
  failures and unnecessary API charges.
- Includes exponential backoff with jitter for retry attempts (up to 3 retries).

---

## Known Security Considerations | 已知安全注意事项

### API Key Handling | API 密钥处理

- LLM API keys are passed to LiteLLM and stored in memory as `SecretStr`.
- If you use the web API's `POST /api/config` endpoint, the API key is
  transmitted in the request body. **Always use HTTPS** in production to protect
  keys in transit.
- API keys are masked (`"***"`) when configs are serialized to YAML on disk,
  but the running process holds the plaintext key in memory.

### CORS Configuration | CORS 配置

- The default CORS policy allows requests from `localhost:8080` only.
- Setting `NUWA_CORS_ORIGINS=*` disables credential forwarding and should never
  be used in production.
- The CORS middleware applies to **all** routes, including the SSE event stream.

### Rate Limiting Scope | 速率限制范围

- Rate limiting is in-memory and per-IP. It does not persist across server
  restarts.
- In a reverse-proxy setup, the client IP may be the proxy's IP. Configure your
  proxy to forward the real client IP via `X-Forwarded-For` or similar headers,
  and consider adding a reverse-proxy-level rate limiter for production use.

### No TLS Termination | 无 TLS 终止

- The built-in uvicorn server does not configure TLS. In production, place Nuwa
  behind a reverse proxy (nginx, Caddy, etc.) that handles HTTPS.

### Authentication Granularity | 认证粒度

- `NUWA_API_KEY` is a single shared secret. There is no per-user or per-role
  authentication. For multi-user deployments, use a reverse proxy with proper
  authentication middleware.
