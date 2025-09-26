# AI Facultative Reinsurance System API Documentation

## Overview

The AI Facultative Reinsurance System API provides comprehensive endpoints for managing facultative reinsurance applications through an AI-powered decision support system.

## Base URL

- Development: `http://localhost:8005`
- Production: `https://api.reinsurance.example.com`

## API Version

Current version: `v1`
All endpoints are prefixed with `/api/v1`

## Authentication

The API supports two authentication methods:

### 1. JWT Bearer Tokens (Recommended for web applications)

```bash
# Login to get tokens
curl -X POST "/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=your_username&password=your_password"

# Use the access token in subsequent requests
curl -X GET "/api/v1/applications/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### 2. API Keys (Recommended for programmatic access)

```bash
curl -X GET "/api/v1/applications/" \
  -H "X-API-Key: YOUR_API_KEY"
```

## Rate Limiting

- **Rate Limit**: 60 requests per minute per IP address
- **Burst Limit**: 100 requests in a short burst
- **Headers**: Rate limit information is returned in response headers:
  - `X-RateLimit-Limit`: Maximum requests per minute
  - `X-RateLimit-Remaining`: Remaining requests in current window
  - `X-RateLimit-Reset`: Unix timestamp when the rate limit resets

## Response Format

All API responses follow a consistent format:

### Success Response
```json
{
  "data": { ... },
  "message": "Success message (optional)",
  "timestamp": "2024-01-15T12:00:00Z"
}
```

### Error Response
```json
{
  "error": {
    "code": 400,
    "message": "Error description",
    "type": "validation_error",
    "details": { ... }
  },
  "request_id": "req_123456789"
}
```

## Endpoints

### Authentication

#### POST /api/v1/auth/login
Login and get access tokens.

**Request:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### POST /api/v1/auth/refresh
Refresh access token using refresh token.

#### GET /api/v1/auth/me
Get current user information.

#### POST /api/v1/auth/logout
Logout (client-side token invalidation).

### Document Management

#### POST /api/v1/documents/upload
Upload a document for processing.

**Supported formats:**
- PDF files (text-based and scanned)
- Email files (.msg, .eml)
- Excel files (.xlsx, .xls, .xlsm)
- Word documents (.docx)
- Images (.jpg, .png) for OCR

**Request:**
```bash
curl -X POST "/api/v1/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf" \
  -F "application_id=app_123"
```

**Response:**
```json
{
  "id": "doc_123",
  "filename": "document.pdf",
  "document_type": "pdf",
  "processed": false,
  "upload_timestamp": "2024-01-15T12:00:00Z",
  "metadata": {}
}
```

#### GET /api/v1/documents/
List documents with filtering and pagination.

**Query Parameters:**
- `application_id`: Filter by application ID
- `skip`: Number of documents to skip (pagination)
- `limit`: Maximum number of documents to return (1-1000)

#### GET /api/v1/documents/{document_id}
Get document metadata by ID.

#### GET /api/v1/documents/{document_id}/download
Download document file content.

#### DELETE /api/v1/documents/{document_id}
Delete document and its associated file.

#### POST /api/v1/documents/{document_id}/ocr
Trigger OCR processing for a document.

#### GET /api/v1/documents/{document_id}/ocr/status
Get OCR processing status and results.

### Application Management

#### POST /api/v1/applications/
Create a new reinsurance application.

#### GET /api/v1/applications/
List applications with filtering, searching, and pagination.

**Query Parameters:**
- `page`: Page number (1-based)
- `page_size`: Items per page (1-100)
- `status`: Filter by application status
- `search`: Search term for application data
- `sort_by`: Field to sort by
- `sort_order`: Sort order (asc/desc)

#### GET /api/v1/applications/{application_id}
Get detailed application information.

#### PUT /api/v1/applications/{application_id}
Update application information.

#### DELETE /api/v1/applications/{application_id}
Delete an application.

#### POST /api/v1/applications/{application_id}/process
Trigger AI processing for an application.

#### GET /api/v1/applications/{application_id}/status
Get detailed processing status for an application.

#### POST /api/v1/applications/{application_id}/approve
Manually approve an application.

#### POST /api/v1/applications/{application_id}/reject
Manually reject an application.

### Analytics and Reporting

#### GET /api/v1/analytics/dashboard
Get key dashboard metrics.

**Response:**
```json
{
  "total_applications": 1250,
  "applications_this_period": 78,
  "approval_rate": 0.72,
  "avg_processing_time_hours": 2.3,
  "total_insured_value": 2500000000.0,
  "high_risk_applications": 15,
  "pending_reviews": 8,
  "system_uptime_percent": 99.8
}
```

#### GET /api/v1/analytics/trends/{metric_type}
Get trend analysis for specific metrics.

**Metric Types:**
- `applications`: Application volume trends
- `approvals`: Approval rate trends
- `rejections`: Rejection rate trends
- `processing_time`: Processing time trends
- `risk_scores`: Risk score distribution trends

#### GET /api/v1/analytics/risk-distribution
Get distribution of applications by risk level.

#### GET /api/v1/analytics/geographic-analysis
Get geographic distribution and analysis.

#### GET /api/v1/analytics/industry-analysis
Get industry sector analysis and trends.

### Business Limits

#### GET /api/v1/business-limits/
Get all business limits and constraints.

#### POST /api/v1/business-limits/
Create a new business limit.

#### PUT /api/v1/business-limits/{limit_id}
Update a business limit.

#### DELETE /api/v1/business-limits/{limit_id}
Delete a business limit.

#### POST /api/v1/business-limits/validate
Validate an application against business limits.

### Task Monitoring

#### GET /api/v1/tasks/status/{task_id}
Get status of a specific Celery task.

#### GET /api/v1/tasks/queue-status
Get status of all processing queues.

#### POST /api/v1/tasks/retry/{task_id}
Retry a failed task.

#### POST /api/v1/tasks/cancel/{task_id}
Cancel a running task.

### System Administration

#### GET /api/v1/admin/health
Get comprehensive system health status.

#### GET /api/v1/admin/users
List all system users.

#### POST /api/v1/admin/users
Create a new user account.

#### GET /api/v1/admin/config
Get system configuration settings.

#### PUT /api/v1/admin/config
Update system configuration.

#### GET /api/v1/admin/logs
Retrieve system logs with filtering.

#### GET /api/v1/admin/audit-log
Retrieve audit log entries.

#### GET /api/v1/admin/backups
List available system backups.

#### POST /api/v1/admin/backups
Create a new system backup.

### Health and Monitoring

#### GET /health
Basic health check (no authentication required).

#### GET /api/v1/health/detailed
Comprehensive health check with dependency testing.

#### GET /api/v1/health/readiness
Kubernetes readiness probe endpoint.

#### GET /api/v1/health/liveness
Kubernetes liveness probe endpoint.

#### GET /api/v1/health/metrics
Prometheus-compatible metrics endpoint.

#### GET /api/v1/health/version
Get application version and build information.

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request - Invalid input or parameters |
| 401 | Unauthorized - Authentication required |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 409 | Conflict - Resource already exists |
| 422 | Unprocessable Entity - Validation error |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - System maintenance |

## Security Features

### Input Validation
- All inputs are validated and sanitized
- Protection against SQL injection and XSS attacks
- File type and size validation for uploads

### Security Headers
All responses include security headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`
- `Content-Security-Policy: ...`

### CORS Policy
CORS is configured to allow requests from:
- `http://localhost:3000` (development)
- `https://reinsurance.example.com` (production)

## Pagination

List endpoints support pagination with the following parameters:

- `page`: Page number (1-based, default: 1)
- `page_size`: Items per page (1-100, default: 20)

Response includes pagination metadata:
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total": 100,
    "has_next": true,
    "has_previous": false
  }
}
```

## Filtering and Searching

Many endpoints support filtering and searching:

### Common Filter Parameters
- `status`: Filter by status
- `date_from`: Filter by date range (start)
- `date_to`: Filter by date range (end)
- `search`: Full-text search

### Example
```bash
curl -X GET "/api/v1/applications/?status=pending&search=technology&page=1&page_size=10" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Webhooks (Future Feature)

The API will support webhooks for real-time notifications:

- Application status changes
- Processing completion
- System alerts
- User actions

## SDKs and Libraries

Official SDKs are planned for:
- Python
- JavaScript/TypeScript
- Java
- C#

## Support and Contact

- **Documentation**: `/docs` (Swagger UI)
- **Alternative Docs**: `/redoc` (ReDoc)
- **OpenAPI Schema**: `/api/openapi.json`
- **Support Email**: support@reinsurance.example.com
- **Status Page**: https://status.reinsurance.example.com

## Changelog

### Version 1.0.0 (2024-01-15)
- Initial API release
- Authentication and authorization
- Document management
- Application processing
- Analytics and reporting
- System administration
- Health monitoring

## Examples

### Complete Application Processing Workflow

```bash
# 1. Login
TOKEN=$(curl -s -X POST "/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=underwriter&password=secret" | jq -r '.access_token')

# 2. Create application
APP_ID=$(curl -s -X POST "/api/v1/applications/" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "pending"}' | jq -r '.id')

# 3. Upload document
DOC_ID=$(curl -s -X POST "/api/v1/documents/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@ri_slip.pdf" \
  -F "application_id=$APP_ID" | jq -r '.id')

# 4. Trigger processing
curl -X POST "/api/v1/applications/$APP_ID/process" \
  -H "Authorization: Bearer $TOKEN"

# 5. Check status
curl -X GET "/api/v1/applications/$APP_ID/status" \
  -H "Authorization: Bearer $TOKEN"

# 6. Get results
curl -X GET "/api/v1/applications/$APP_ID" \
  -H "Authorization: Bearer $TOKEN"
```

### Batch Document Processing

```bash
# Upload multiple documents
for file in *.pdf; do
  curl -X POST "/api/v1/documents/upload" \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@$file" \
    -F "application_id=$APP_ID"
done

# Trigger batch OCR processing
curl -X POST "/api/v1/documents/batch-ocr" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_ids": ["doc_1", "doc_2", "doc_3"]}'
```

### Analytics Dashboard Data

```bash
# Get dashboard metrics
curl -X GET "/api/v1/analytics/dashboard?time_range=30d" \
  -H "Authorization: Bearer $TOKEN"

# Get trend data
curl -X GET "/api/v1/analytics/trends/applications?granularity=daily" \
  -H "Authorization: Bearer $TOKEN"

# Get risk distribution
curl -X GET "/api/v1/analytics/risk-distribution" \
  -H "Authorization: Bearer $TOKEN"
```