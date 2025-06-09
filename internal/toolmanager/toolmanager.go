package toolmanager

import (
	"context"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/alextheberge/opsy/assets"
	"github.com/alextheberge/opsy/internal/agent"
	"github.com/alextheberge/opsy/internal/config"
	"github.com/alextheberge/opsy/internal/tool"
	"gopkg.in/yaml.v3"
)

const (
	// ErrLoadingTools is the error message for failed to load tools.
	ErrLoadingTools = "failed to load tools"
	// ErrLoadingTool is the error message for failed to load a specific tool.
	ErrLoadingTool = "failed to load tool"
	// ErrParsingTool is the error message for failed to parse a tool.
	ErrParsingTool = "failed to parse tool"
	// ErrToolNotFound is the error message for a tool not found.
	ErrToolNotFound = "tool not found"
	// ErrInvalidToolDefinition is the error message for an invalid tool definition.
	ErrInvalidToolDefinition = "invalid tool definition"
)

// Manager is the interface for the tool manager.
type Manager interface {
	// LoadTools loads the tools from the tool manager.
	LoadTools() error
	// GetTools returns all tools.
	GetTools() map[string]tool.Tool
	// GetTool returns a tool by name.
	GetTool(name string) (tool.Tool, error)
}

// ToolManager is the tool manager.
type ToolManager struct {
	cfg    config.Configuration
	logger *slog.Logger
	ctx    context.Context
	fs     fs.FS
	dir    string
	tools  map[string]tool.Tool
	agent  *agent.Agent
	mu     sync.RWMutex
}

// Option is a function that modifies the tool manager.
type Option func(*ToolManager)

// New creates a new tool manager.
func New(opts ...Option) *ToolManager {
	tm := &ToolManager{
		cfg:    config.New().GetConfig(),
		logger: slog.New(slog.DiscardHandler),
		ctx:    context.Background(),
		fs:     assets.Tools,
		dir:    assets.ToolsDir,
		tools:  make(map[string]tool.Tool),
		agent:  nil,
	}

	for _, opt := range opts {
		opt(tm)
	}

	tm.logger.WithGroup("config").With("directory", tm.dir).Debug("Tool manager initialized.")

	return tm
}

// WithConfig sets the configuration for the tool manager.
func WithConfig(cfg config.Configuration) Option {
	return func(tm *ToolManager) {
		tm.cfg = cfg
	}
}

// WithAgent sets the agent for the tool manager.
func WithAgent(agent *agent.Agent) Option {
	return func(tm *ToolManager) {
		tm.agent = agent
	}
}

// WithLogger sets the logger for the tool manager.
func WithLogger(logger *slog.Logger) Option {
	return func(tm *ToolManager) {
		tm.logger = logger.With("component", "toolmanager")
	}
}

// WithDirectory sets the directory for the tool manager.
func WithDirectory(dir string) Option {
	return func(tm *ToolManager) {
		tm.fs = os.DirFS(dir)
		tm.dir = "."
	}
}

// WithContext sets the context for the tool manager.
func WithContext(ctx context.Context) Option {
	return func(tm *ToolManager) {
		tm.ctx = ctx
	}
}

// LoadTools loads the tools from the tool manager.
func (tm *ToolManager) LoadTools() error {
	toolFiles, err := fs.ReadDir(tm.fs, tm.dir)
	if err != nil {
		return fmt.Errorf("%s: %v", ErrLoadingTools, err)
	}

	tm.mu.Lock()
	defer tm.mu.Unlock()

	for k := range tm.tools {
		delete(tm.tools, k)
	}

	// Exec tool is a special tool which we always statically load.
	tm.tools[tool.ExecToolName] = tool.NewExecTool(tm.logger, &tm.cfg.Tools)

	for _, toolFile := range toolFiles {
		if toolFile.IsDir() {
			continue
		}

		name := strings.TrimSuffix(toolFile.Name(), filepath.Ext(toolFile.Name()))
		tool, err := tm.loadTool(name, toolFile)
		if err != nil {
			tm.logger.With("tool.name", name).With("filename", toolFile.Name()).With("error", err).
				Error("Failed to load the tool.")
			continue
		}

		tm.tools[name] = tool
	}

	tm.logger.With("tools.count", len(tm.tools)).Debug("Tools loaded.")

	return nil
}

// loadTool loads a tool from a file.
func (tm *ToolManager) loadTool(name string, toolFile fs.DirEntry) (tool.Tool, error) {
	contents, err := fs.ReadFile(tm.fs, filepath.Join(tm.dir, toolFile.Name()))
	if err != nil {
		return nil, fmt.Errorf("%s: %v", ErrLoadingTool, err)
	}

	var definition tool.Definition
	if err := yaml.Unmarshal(contents, &definition); err != nil {
		return nil, fmt.Errorf("%s: %v", ErrParsingTool, err)
	}

	if err := tool.ValidateDefinition(&definition); err != nil {
		return nil, fmt.Errorf("%s: %s: %v", ErrInvalidToolDefinition, name, err)
	}

	return tool.New(name, definition, tm.logger, &tm.cfg.Tools, tm.agent), nil
}

// GetTools returns all tools.
func (tm *ToolManager) GetTools() map[string]tool.Tool {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	return tm.tools
}

// GetTool returns a tool by name.
func (tm *ToolManager) GetTool(name string) (tool.Tool, error) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	tool, ok := tm.tools[name]
	if !ok {
		return nil, fmt.Errorf("%s: %v", ErrToolNotFound, name)
	}

	return tool, nil
}
