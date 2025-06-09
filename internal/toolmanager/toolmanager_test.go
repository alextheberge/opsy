package toolmanager

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/alextheberge/opsy/internal/agent"
	"github.com/alextheberge/opsy/internal/config"
	"github.com/alextheberge/opsy/internal/tool"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newTestAgent creates a new agent for testing
func newTestAgent() *agent.Agent {
	return agent.New(
		agent.WithConfig(config.New().GetConfig()),
		agent.WithLogger(slog.New(slog.DiscardHandler)),
	)
}

// TestNew tests the creation of a new tool manager with various options.
func TestNew(t *testing.T) {
	t.Run("creates default tool manager", func(t *testing.T) {
		tm := New()
		assert.NotNil(t, tm)
		assert.Equal(t, "tools", tm.dir)
		assert.NotNil(t, tm.tools)
		assert.Nil(t, tm.agent)
	})

	t.Run("creates tool manager with options", func(t *testing.T) {
		logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
		cfg := config.New().GetConfig()
		ctx := context.Background()
		agent := newTestAgent()

		tm := New(
			WithLogger(logger),
			WithConfig(cfg),
			WithContext(ctx),
			WithDirectory("testdata"),
			WithAgent(agent),
		)

		assert.NotNil(t, tm)
		assert.Equal(t, ".", tm.dir)
		assert.Equal(t, cfg, tm.cfg)
		assert.Equal(t, ctx, tm.ctx)
		assert.Equal(t, agent, tm.agent)
	})
}

// TestLoadTools tests loading tools from the filesystem.
func TestLoadTools(t *testing.T) {
	t.Run("loads valid tools", func(t *testing.T) {
		tm := New(
			WithDirectory("testdata"),
			WithAgent(newTestAgent()),
		)
		err := tm.LoadTools()
		require.NoError(t, err)

		tools := tm.GetTools()
		assert.Len(t, tools, 3) // Should load test_tool.yaml, executable_tool.yaml and exec tool

		tl, ok := tools["test_tool"]
		require.True(t, ok)
		assert.Equal(t, "Test Tool", tl.GetDisplayName())
		assert.Equal(t, "A tool for testing purposes", tl.GetDescription())

		// Verify executable tool is loaded
		executableTool, ok := tools["executable_tool"]
		require.True(t, ok)
		assert.Equal(t, "Executable Tool", executableTool.GetDisplayName())
		assert.Equal(t, "A test tool with executable", executableTool.GetDescription())

		// Verify exec tool is loaded
		execTool, ok := tools[tool.ExecToolName]
		require.True(t, ok)
		assert.Equal(t, "Exec", execTool.GetDisplayName())
	})

	t.Run("handles invalid tool definitions", func(t *testing.T) {
		// The invalid tool should be skipped during loading
		tm := New(
			WithDirectory("testdata"),
			WithAgent(newTestAgent()),
		)
		err := tm.LoadTools()
		require.NoError(t, err)

		_, err = tm.GetTool("invalid_tool")
		assert.ErrorContains(t, err, ErrToolNotFound)
	})

	t.Run("validates required fields", func(t *testing.T) {
		def := &tool.Definition{
			DisplayName: "Test Tool",
			Description: "Test Description",
			Inputs: map[string]tool.Input{
				"test_input": {
					Description: "Test Input", // Missing type
				},
			},
		}
		err := tool.ValidateDefinition(def)
		assert.ErrorContains(t, err, fmt.Sprintf("%s: %q", tool.ErrToolInputMissingType, "test_input"))

		def.Inputs["test_input"] = tool.Input{
			Type: "string", // Missing description
		}
		err = tool.ValidateDefinition(def)
		assert.ErrorContains(t, err, fmt.Sprintf("%s: %q", tool.ErrToolInputMissingDescription, "test_input"))

		def.DisplayName = ""
		err = tool.ValidateDefinition(def)
		assert.ErrorContains(t, err, tool.ErrToolMissingDisplayName)

		def.DisplayName = "Test Tool"
		def.Description = ""
		err = tool.ValidateDefinition(def)
		assert.ErrorContains(t, err, tool.ErrToolMissingDescription)
	})

	t.Run("handles non-existent directory", func(t *testing.T) {
		tm := New(WithDirectory("nonexistent"))
		err := tm.LoadTools()
		assert.ErrorContains(t, err, ErrLoadingTools)
	})

	t.Run("handles non-yaml files", func(t *testing.T) {
		// Create a temporary directory for test files
		tmpDir := t.TempDir()

		// Create a non-YAML file
		err := os.WriteFile(filepath.Join(tmpDir, "not_a_tool.txt"), []byte("not a tool"), 0644)
		require.NoError(t, err)

		// Create a valid YAML file
		err = os.WriteFile(filepath.Join(tmpDir, "valid_tool.yaml"), []byte(`
display_name: "Valid Tool"
description: "A valid tool"
inputs:
  test_input:
    type: "string"
    description: "Test input"
`), 0644)
		require.NoError(t, err)

		tm := New(WithDirectory(tmpDir))
		err = tm.LoadTools()
		require.NoError(t, err)

		tools := tm.GetTools()
		assert.Len(t, tools, 2) // Should only load valid_tool.yaml and exec tool
	})

	t.Run("handles empty directory", func(t *testing.T) {
		tmpDir := t.TempDir()
		tm := New(
			WithDirectory(tmpDir),
			WithAgent(newTestAgent()),
		)
		err := tm.LoadTools()
		require.NoError(t, err)
		assert.Len(t, tm.GetTools(), 1) // Should only have exec tool
	})

	t.Run("handles directory with only invalid tools", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Create an invalid YAML file
		err := os.WriteFile(filepath.Join(tmpDir, "invalid.yaml"), []byte("invalid: yaml: content"), 0644)
		require.NoError(t, err)

		tm := New(
			WithDirectory(tmpDir),
			WithAgent(newTestAgent()),
		)
		err = tm.LoadTools()
		require.NoError(t, err)
		assert.Len(t, tm.GetTools(), 1) // Should only have exec tool
	})

	t.Run("handles invalid executable path", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Create a tool with invalid executable path
		err := os.WriteFile(filepath.Join(tmpDir, "invalid_exec.yaml"), []byte(`
display_name: "Invalid Exec Tool"
description: "A tool with invalid executable"
executable: "/nonexistent/path"
inputs:
  test_input:
    type: "string"
    description: "Test input"
`), 0644)
		require.NoError(t, err)

		tm := New(
			WithDirectory(tmpDir),
			WithAgent(newTestAgent()),
		)
		err = tm.LoadTools()
		require.NoError(t, err)
		assert.Len(t, tm.GetTools(), 1) // Should only have exec tool
	})

	t.Run("handles_invalid_system_prompt", func(t *testing.T) {
		// Create a temporary directory for the test
		dir := t.TempDir()
		require.NoError(t, os.WriteFile(filepath.Join(dir, "invalid_prompt.yaml"), []byte(`
display_name: Invalid Prompt Tool
description: A tool with an invalid system prompt
rules:
  - "valid rule"
executable: /path/to/nonexistent/executable # This will cause system prompt validation to fail
inputs:
  input1:
    type: string
    description: Test input
`), 0644))

		// Create a test logger that captures logs
		var logBuffer strings.Builder
		logger := slog.New(slog.NewTextHandler(&logBuffer, &slog.HandlerOptions{
			Level: slog.LevelError,
		}))

		// Create a tool manager with the test directory and logger
		tm := New(WithDirectory(dir), WithLogger(logger))

		// Load tools
		require.NoError(t, tm.LoadTools())

		// Verify that the error was logged
		require.Contains(t, logBuffer.String(), ErrInvalidToolDefinition)
		require.Contains(t, logBuffer.String(), "invalid_prompt")

		// Verify that getting the invalid tool returns an error
		_, err := tm.GetTool("invalid_prompt")
		require.Error(t, err)
		require.Contains(t, err.Error(), ErrToolNotFound)
	})
}

// TestGetTool tests retrieving specific tools.
func TestGetTool(t *testing.T) {
	tm := New(
		WithDirectory("testdata"),
		WithAgent(newTestAgent()),
	)
	require.NoError(t, tm.LoadTools())

	t.Run("gets existing tool", func(t *testing.T) {
		tool, err := tm.GetTool("test_tool")
		require.NoError(t, err)
		assert.Equal(t, "Test Tool", tool.GetDisplayName())
	})

	t.Run("gets exec tool", func(t *testing.T) {
		tool, err := tm.GetTool(tool.ExecToolName)
		require.NoError(t, err)
		assert.Equal(t, "Exec", tool.GetDisplayName())
	})

	t.Run("returns error for non-existent tool", func(t *testing.T) {
		_, err := tm.GetTool("nonexistent")
		assert.ErrorContains(t, err, ErrToolNotFound)
	})
}

// TestGetTools tests retrieving all tools.
func TestGetTools(t *testing.T) {
	tm := New(
		WithDirectory("testdata"),
		WithAgent(newTestAgent()),
	)
	require.NoError(t, tm.LoadTools())

	tools := tm.GetTools()
	assert.Len(t, tools, 3) // Should have test_tool, executable_tool and exec tool

	// Verify test_tool
	testTool, ok := tools["test_tool"]
	require.True(t, ok, "test_tool should be present")
	assert.Equal(t, "Test Tool", testTool.GetDisplayName())

	// Verify executable_tool
	executableTool, ok := tools["executable_tool"]
	require.True(t, ok, "executable_tool should be present")
	assert.Equal(t, "Executable Tool", executableTool.GetDisplayName())

	// Verify exec tool
	execTool, ok := tools[tool.ExecToolName]
	require.True(t, ok, "exec tool should be present")
	assert.Equal(t, "Exec", execTool.GetDisplayName())
}

// TestConcurrentAccess tests thread safety of the tool manager.
func TestConcurrentAccess(t *testing.T) {
	tm := New(
		WithDirectory("testdata"),
		WithAgent(newTestAgent()),
	)
	require.NoError(t, tm.LoadTools())

	var wg sync.WaitGroup
	numGoroutines := 10

	// Test concurrent tool access
	wg.Add(numGoroutines)
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			tools := tm.GetTools()
			assert.NotEmpty(t, tools)
			tool, err := tm.GetTool("test_tool")
			assert.NoError(t, err)
			assert.Equal(t, "Test Tool", tool.GetDisplayName())
		}()
	}
	wg.Wait()

	// Test concurrent tool loading and access
	wg.Add(numGoroutines * 2)
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			err := tm.LoadTools()
			assert.NoError(t, err)
		}()
		go func() {
			defer wg.Done()
			tools := tm.GetTools()
			assert.NotEmpty(t, tools)
		}()
	}
	wg.Wait()
}
