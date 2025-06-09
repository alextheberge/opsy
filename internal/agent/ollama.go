package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// OllamaClient is a client for the Ollama API.
type OllamaClient struct {
	host       string
	httpClient *http.Client
}

// NewOllamaClient creates a new Ollama client.
func NewOllamaClient(host string) *OllamaClient {
	return &OllamaClient{
		host: host,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

// OllamaRequest represents a request to the Ollama API.
type OllamaRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	System      string  `json:"system,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	MaxTokens   int64   `json:"max_tokens,omitempty"`
	Stream      bool    `json:"stream,omitempty"`
}

// OllamaResponse represents a response from the Ollama API.
type OllamaResponse struct {
	Model     string `json:"model"`
	Response  string `json:"response"`
	CreatedAt string `json:"created_at"`
	Done      bool   `json:"done"`
}

// Generate generates a response from the Ollama API.
func (c *OllamaClient) Generate(ctx context.Context, req OllamaRequest) (*OllamaResponse, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.host+"/api/generate", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d, body: %s", resp.StatusCode, string(body))
	}

	var response OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}
