package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/alextheberge/opsy/assets"
	"github.com/alextheberge/opsy/internal/config"
	"github.com/alextheberge/opsy/internal/tool"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
)

const (
	// ErrNoRunOptions is the error returned when no run options are provided.
	ErrNoRunOptions = "no run options provided"
	// ErrNoTaskProvided is the error returned when no task is provided.
	ErrNoTaskProvided = "no task provided"

	// StatusReady is the status of the agent when it is ready to run.
	StatusReady = "Ready"
	// StatusRunning is the status of the agent when it is running.
	StatusRunning = "Running"
	// StatusFinished is the status of the agent when it has finished.
	StatusFinished = "Finished"
	// StatusError is the status of the agent when it has encountered an error.
	StatusError = "Error"
)

// Status is the status of the agent.
type Status string

// Agent is a struct that contains the state of the agent.
type Agent struct {
	anthropicClient *anthropic.Client
	ollamaClient    *OllamaClient
	ctx             context.Context
	cfg             config.Configuration
	logger          *slog.Logger
	communication   *Communication
}

// Message is a struct that contains a message from the agent.
type Message struct {
	// Tool is the name of the tool that sent the message.
	Tool string
	// Message is the message from the tool.
	Message string
	// Timestamp is the timestamp when the message was sent.
	Timestamp time.Time
}

// Communication is a struct that contains the communication channels for the agent.
type Communication struct {
	Commands chan tool.Command
	Messages chan Message
	Status   chan Status
}

// Option is a function that configures the Agent.
type Option func(*Agent)

const (
	// Name is the name of the agent.
	Name = "Opsy"
)

// New creates a new Agent.
func New(opts ...Option) *Agent {
	a := &Agent{
		ctx:    context.Background(),
		cfg:    config.New().GetConfig(),
		logger: slog.New(slog.DiscardHandler),
		communication: &Communication{
			Commands: make(chan tool.Command),
			Messages: make(chan Message),
			Status:   make(chan Status),
		},
	}

	for _, opt := range opts {
		opt(a)
	}

	// Initialize the appropriate client based on the configuration
	if a.cfg.Ollama.Enabled {
		a.ollamaClient = NewOllamaClient(a.cfg.Ollama.Host)
		a.logger.WithGroup("config").With("max_tokens", a.cfg.Ollama.MaxTokens).With("model", a.cfg.Ollama.Model).
			With("temperature", a.cfg.Ollama.Temperature).With("host", a.cfg.Ollama.Host).Debug("Agent initialized with Ollama.")
	} else if a.cfg.Anthropic.APIKey != "" {
		c := anthropic.NewClient(option.WithAPIKey(a.cfg.Anthropic.APIKey))
		a.anthropicClient = &c
		a.logger.WithGroup("config").With("max_tokens", a.cfg.Anthropic.MaxTokens).With("model", a.cfg.Anthropic.Model).
			With("temperature", a.cfg.Anthropic.Temperature).Debug("Agent initialized with Anthropic.")
	}

	return a
}

// WithContext sets the context for the agent.
func WithContext(ctx context.Context) Option {
	return func(a *Agent) {
		a.ctx = ctx
	}
}

// WithConfig sets the configuration for the agent.
func WithConfig(cfg config.Configuration) Option {
	return func(a *Agent) {
		a.cfg = cfg
	}
}

// WithLogger sets the logger for the agent.
func WithLogger(logger *slog.Logger) Option {
	return func(a *Agent) {
		a.logger = logger.With("component", "agent")
	}
}

// WithClient sets the Anthropic client for the agent.
func WithClient(client *anthropic.Client) Option {
	return func(a *Agent) {
		a.anthropicClient = client
	}
}

// WithOllamaClient sets the Ollama client for the agent.
func WithOllamaClient(client *OllamaClient) Option {
	return func(a *Agent) {
		a.ollamaClient = client
	}
}

// WithCommunication sets the communication channels for the agent.
func WithCommunication(communication *Communication) Option {
	return func(a *Agent) {
		a.communication = communication
	}
}

// Run runs the agent with the given task and tools.
func (a *Agent) Run(opts *tool.RunOptions, ctx context.Context) ([]tool.Output, error) {
	if opts == nil {
		return nil, errors.New(ErrNoRunOptions)
	}

	if opts.Task == "" {
		return nil, errors.New(ErrNoTaskProvided)
	}

	if ctx == nil {
		ctx = a.ctx
	}

	prompt, err := assets.RenderAgentSystemPrompt(&assets.AgentSystemPromptData{
		Shell: a.cfg.Tools.Exec.Shell,
	})
	if err != nil {
		return nil, fmt.Errorf("%s: %w", assets.ErrToolRenderingPrompt, err)
	}

	if opts.Prompt != "" {
		prompt = opts.Prompt
	}

	logger := a.logger.With("task", opts.Task).With("tool", opts.Caller).With("tools.count", len(opts.Tools))
	logger.Debug("Agent running.")
	a.communication.Status <- StatusRunning

	// Use Ollama if enabled, otherwise use Anthropic
	if a.cfg.Ollama.Enabled {
		return a.runWithOllama(opts, ctx, prompt, logger)
	} else {
		return a.runWithAnthropic(opts, ctx, prompt, logger)
	}
}

// runWithOllama runs the agent with the Ollama client.
func (a *Agent) runWithOllama(opts *tool.RunOptions, ctx context.Context, prompt string, logger *slog.Logger) ([]tool.Output, error) {
	var output []tool.Output

	// Format the task and tools for Ollama
	toolsDescription := ""
	if len(opts.Tools) > 0 {
		toolsDescription = "You have access to the following tools:\n\n"
		for _, t := range opts.Tools {
			toolsDescription += fmt.Sprintf("- %s: %s\n", t.GetName(), t.GetDescription())
		}
		toolsDescription += "\nWhen you need to use a tool, respond with a JSON object in the following format:\n"
		toolsDescription += "```json\n{\"tool\": \"tool_name\", \"input\": {\"param1\": \"value1\", \"param2\": \"value2\"}}\n```\n"
		toolsDescription += "After receiving the tool output, continue the conversation."
	}

	// Combine the system prompt with tools description
	systemPrompt := prompt
	if toolsDescription != "" {
		systemPrompt += "\n\n" + toolsDescription
	}

	// Create the initial request
	req := OllamaRequest{
		Model:       a.cfg.Ollama.Model,
		Prompt:      opts.Task,
		System:      systemPrompt,
		Temperature: a.cfg.Ollama.Temperature,
		MaxTokens:   a.cfg.Ollama.MaxTokens,
	}

	// Keep track of the conversation history
	conversation := opts.Task

	for {
		// Send the request to Ollama
		resp, err := a.ollamaClient.Generate(ctx, req)
		if err != nil {
			logger.With("error", err).Error("Failed to send message to Ollama API.")
			return nil, err
		}

		// Process the response
		response := resp.Response

		// Check if the response contains a tool call
		toolCallStart := strings.Index(response, "{\"tool\":")
		if toolCallStart == -1 {
			// No tool call, just return the response
			a.communication.Messages <- Message{
				Tool:      opts.Caller,
				Message:   response,
				Timestamp: time.Now(),
			}

			// Add the final output
			output = append(output, tool.Output{
				Result: response,
			})

			break
		}

		// Extract the tool call JSON
		toolCallEnd := strings.LastIndex(response, "}") + 1
		toolCallJSON := response[toolCallStart:toolCallEnd]

		// Parse the tool call
		var toolCall struct {
			Tool  string                 `json:"tool"`
			Input map[string]interface{} `json:"input"`
		}

		if err := json.Unmarshal([]byte(toolCallJSON), &toolCall); err != nil {
			logger.With("error", err).Error("Failed to parse tool call.")
			continue
		}

		// Find the tool
		var selectedTool tool.Tool
		for _, t := range opts.Tools {
			if t.GetName() == toolCall.Tool {
				selectedTool = t
				break
			}
		}

		if selectedTool == nil {
			logger.With("tool", toolCall.Tool).Error("Tool not found.")
			continue
		}

		// Execute the tool
		toolOutput, err := selectedTool.Execute(toolCall.Input, ctx)
		if err != nil {
			logger.With("error", err).Error("Failed to run tool.")
			toolOutput = &tool.Output{
				Result: fmt.Sprintf("Error: %v", err),
				IsError: true,
			}
		}

		// Add the tool output to the conversation
		toolOutputText := fmt.Sprintf("\nTool: %s\nInput: %v\nOutput: %s\n",
			toolCall.Tool,
			toolCall.Input,
			toolOutput.Result)

		// Update the conversation and prompt for the next request
		conversation += "\n" + response + toolOutputText
		req.Prompt = conversation

		// Add the tool output to the output
		output = append(output, *toolOutput)

		// Send the tool output as a message
		a.communication.Messages <- Message{
			Tool:      toolCall.Tool,
			Message:   toolOutput.Result,
			Timestamp: time.Now(),
		}
	}

	return output, nil
}

// runWithAnthropic runs the agent with the Anthropic client.
func (a *Agent) runWithAnthropic(opts *tool.RunOptions, ctx context.Context, prompt string, logger *slog.Logger) ([]tool.Output, error) {
	var output []tool.Output
	messages := []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(opts.Task))}

	for {
		msg := anthropic.MessageNewParams{
			Model:       a.cfg.Anthropic.Model,
			MaxTokens:   a.cfg.Anthropic.MaxTokens,
			System:      []anthropic.TextBlockParam{{Text: prompt}},
			Messages:    messages,
			Tools:       convertTools(opts.Tools),
			Temperature: param.NewOpt(a.cfg.Anthropic.Temperature),
		}

		if len(opts.Tools) > 0 {
			msg.ToolChoice = anthropic.ToolChoiceUnionParam{
				OfToolChoiceAuto: &anthropic.ToolChoiceAutoParam{
					DisableParallelToolUse: param.NewOpt(true),
				},
			}
		}

		message, err := a.anthropicClient.Messages.New(ctx, msg)

		if err != nil {
			// TODO(t-dabasinskas): Implement retry logic
			logger.With("error", err).Error("Failed to send message to Anthropic API.")
			return nil, err
		}

		toolResults := []anthropic.ContentBlockParamUnion{}
		for _, block := range message.Content {
			switch block.Type {
			case "text":
				a.communication.Messages <- Message{
					Tool:      opts.Caller,
					Message:   block.Text,
					Timestamp: time.Now(),
				}
			case "tool_use":
				isError := false
				resultBlockContent := ""
				toolInputs := map[string]any{}

				if err := json.Unmarshal(block.Input, &toolInputs); err != nil {
					logger.With("error", err).Error("Failed to unmarshal tool inputs.")
					continue
				}

				var toolOutput *tool.Output
				tool, ok := opts.Tools[block.Name]
				if !ok {
					logger.With("tool_name", block.Name).Warn("Tool not found, skipping.")
					continue
				}

				toolOutput, err = tool.Execute(toolInputs, ctx)
				if err != nil {
					logger.With("error", err).Error("Failed to execute tool.")
					isError = true
				}

				if toolOutput == nil {
					logger.With("tool_name", block.Name).Warn("Tool has no output, skipping.")
					continue
				}

				output = append(output, *toolOutput)

				// Handle messages from all the tools except the Exec:
				if toolOutput.Result != "" && toolOutput.ExecutedCommand == nil {
					resultBlockContent = toolOutput.Result
					a.communication.Messages <- Message{
						Tool:      opts.Caller,
						Message:   toolOutput.Result,
						Timestamp: time.Now(),
					}
				}
				logger.With("output", toolOutput).Warn(">>>>Tool result.")

				// Handle messages from the Exec tool:
				if toolOutput.ExecutedCommand != nil {
					resultBlockContent = toolOutput.ExecutedCommand.Output
					isError = toolOutput.ExecutedCommand.ExitCode != 0
					a.communication.Commands <- *toolOutput.ExecutedCommand
				}

				resultBlock := anthropic.NewToolResultBlock(block.ID, resultBlockContent, isError)
				toolResults = append(toolResults, resultBlock)
			}
		}

		messages = append(messages, message.ToParam())
		if len(toolResults) == 0 {
			break
		}

		messages = append(messages, anthropic.NewUserMessage(toolResults...))
	}

	return output, nil
}

// convertTools converts the tools to the format required by the Anthropic SDK.
func convertTools(tools map[string]tool.Tool) (anthropicTools []anthropic.ToolUnionParam) {
	for _, t := range tools {
		anthropicTools = append(anthropicTools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        t.GetName(),
				Description: param.NewOpt(t.GetDescription()),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: t.GetInputSchema().Properties,
				},
			},
		})
	}
	return
}
