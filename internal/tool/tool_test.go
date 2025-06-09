package tool

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"testing"

	"github.com/alextheberge/opsy/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockRunner is a mock implementation of the Runner interface for testing.
type mockRunner struct {
	outputs []Output
	err     error
}

func (r *mockRunner) Run(opts *RunOptions, ctx context.Context) ([]Output, error) {
	if r.err != nil {
		return nil, r.err
	}
	return r.outputs, nil
}

// newMockRunner creates a new mock runner with the given outputs and error.
func newMockRunner(outputs []Output, err error) *mockRunner {
	return &mockRunner{
		outputs: outputs,
		err:     err,
	}
}

// TestNewTool tests the creation of a new tool.
func TestNewTool(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(nil, nil))
	cfg := &config.ToolsConfiguration{
		Timeout: 120,
		Exec: config.ExecToolConfiguration{
			Timeout: 60,
			Shell:   "/bin/bash",
		},
	}
	definition := Definition{
		DisplayName: "Test Tool",
		Description: "Test Description",
		Inputs:      make(map[string]Input),
	}

	runner := newMockRunner(nil, nil)
	tool := New("test", definition, logger, cfg, runner)

	t.Run("initializes with correct values", func(t *testing.T) {
		assert.Equal(t, "test", tool.name)
		assert.Equal(t, "Test Tool", tool.GetDisplayName())
		assert.Equal(t, "Test Description", tool.GetDescription())
		assert.NotNil(t, tool.GetInputSchema())
		assert.Equal(t, cfg, tool.config)
		assert.Equal(t, runner, tool.agent)
	})
}

// TestToolGetters tests the getter methods of Tool.
func TestToolGetters(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(nil, nil))
	cfg := &config.ToolsConfiguration{
		Timeout: 120,
		Exec: config.ExecToolConfiguration{
			Timeout: 60,
			Shell:   "/bin/bash",
		},
	}
	inputs := map[string]Input{
		"test_input": {
			Type:        "string",
			Description: "Test Input",
			Default:     "default",
			Examples:    []any{"example"},
			Optional:    false,
		},
	}

	definition := Definition{
		DisplayName: "Display Name",
		Description: "Description",
		Inputs:      inputs,
	}

	runner := newMockRunner(nil, nil)
	tool := New("test", definition, logger, cfg, runner)

	t.Run("GetDisplayName returns correct value", func(t *testing.T) {
		assert.Equal(t, "Display Name", tool.GetDisplayName())
	})

	t.Run("GetDescription returns correct value", func(t *testing.T) {
		assert.Equal(t, "Description", tool.GetDescription())
	})

	t.Run("GetInputSchema returns valid schema", func(t *testing.T) {
		schema := tool.GetInputSchema()
		require.NotNil(t, schema)

		// Verify schema properties
		prop, ok := schema.Properties.Get("test_input")
		require.True(t, ok)
		assert.Equal(t, "string", prop.Type)
		assert.Equal(t, "Test Input", prop.Description)
		assert.Equal(t, "default", prop.Default)
		assert.Equal(t, []any{"example"}, prop.Examples)
	})
}

// TestGenerateInputSchema tests the schema generation for tool inputs.
func TestGenerateInputSchema(t *testing.T) {
	inputs := map[string]Input{
		"required_input": {
			Type:        "string",
			Description: "Required Input",
			Default:     "default",
			Examples:    []any{"example1", "example2"},
			Optional:    false,
		},
		"optional_input": {
			Type:        "string",
			Description: "Optional Input",
			Default:     "",
			Examples:    []any{"optional"},
			Optional:    true,
		},
	}

	schema := generateInputSchema(inputs)

	t.Run("generates correct schema structure", func(t *testing.T) {
		assert.Equal(t, "object", schema.Type)
		assert.Contains(t, schema.Required, "required_input")
		assert.NotContains(t, schema.Required, "optional_input")
	})

	t.Run("includes all inputs in properties", func(t *testing.T) {
		requiredProp, ok := schema.Properties.Get("required_input")
		require.True(t, ok)
		assert.Equal(t, "Required Input", requiredProp.Description)
		assert.Equal(t, "default", requiredProp.Default)
		assert.Equal(t, []any{"example1", "example2"}, requiredProp.Examples)

		optionalProp, ok := schema.Properties.Get("optional_input")
		require.True(t, ok)
		assert.Equal(t, "Optional Input", optionalProp.Description)
		assert.Equal(t, "", optionalProp.Default)
		assert.Equal(t, []any{"optional"}, optionalProp.Examples)
	})
}

// TestToolExecute tests the Execute method of Tool.
func TestToolExecute(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	cfg := &config.ToolsConfiguration{
		Timeout: 120,
		Exec: config.ExecToolConfiguration{
			Timeout: 60,
			Shell:   "/bin/bash",
		},
	}

	t.Run("returns empty result", func(t *testing.T) {
		runner := newMockRunner(nil, nil)
		tool := New("test", Definition{
			DisplayName: "Test Tool",
			Description: "Test Description",
			Inputs:      map[string]Input{},
		}, logger, cfg, runner)

		input := map[string]any{
			"test":                "value",
			inputTask:             "test task",
			inputWorkingDirectory: ".",
		}
		output, err := tool.Execute(input, context.Background())
		require.NoError(t, err)
		assert.NotNil(t, output)
		assert.Equal(t, "Test Tool", output.Tool)
		assert.Empty(t, output.Result)
		assert.False(t, output.IsError)
		assert.Nil(t, output.ExecutedCommand)
	})

	t.Run("handles runner error", func(t *testing.T) {
		expectedErr := errors.New("test error")
		runner := newMockRunner(nil, expectedErr)
		tool := New("test", Definition{
			DisplayName: "Test Tool",
			Description: "Test Description",
			Inputs:      map[string]Input{},
		}, logger, cfg, runner)

		input := map[string]any{
			"test":                "value",
			inputTask:             "test task",
			inputWorkingDirectory: ".",
		}
		output, err := tool.Execute(input, context.Background())
		assert.ErrorIs(t, err, expectedErr)
		assert.NotNil(t, output)
		assert.Equal(t, "Test Tool", output.Tool)
		assert.Empty(t, output.Result)
		assert.True(t, output.IsError)
		assert.Nil(t, output.ExecutedCommand)
	})

	t.Run("returns runner output", func(t *testing.T) {
		expectedOutput := []Output{{
			Tool:   "test",
			Result: "test result",
		}}
		runner := newMockRunner(expectedOutput, nil)
		tool := New("test", Definition{
			DisplayName: "Test Tool",
			Description: "Test Description",
			Inputs:      map[string]Input{},
		}, logger, cfg, runner)

		input := map[string]any{
			"test":                "value",
			inputTask:             "test task",
			inputWorkingDirectory: ".",
		}
		output, err := tool.Execute(input, context.Background())
		require.NoError(t, err)
		assert.NotNil(t, output)
		assert.Equal(t, "Test Tool", output.Tool)
		assert.Equal(t, "test result", output.Result)
		assert.False(t, output.IsError)
		assert.Nil(t, output.ExecutedCommand)
	})

	t.Run("validates task input", func(t *testing.T) {
		runner := newMockRunner(nil, nil)
		tool := New("test", Definition{
			DisplayName: "Test Tool",
			Description: "Test Description",
			Inputs:      map[string]Input{},
		}, logger, cfg, runner)

		input := map[string]any{
			"test":                "value",
			inputTask:             123, // Invalid type
			inputWorkingDirectory: ".",
		}
		output, err := tool.Execute(input, context.Background())
		assert.Error(t, err)
		assert.Contains(t, err.Error(), ErrInvalidToolInputType)
		assert.Nil(t, output)
	})
}

// TestAppendCommonInputs tests the appendCommonInputs function.
func TestAppendCommonInputs(t *testing.T) {
	customInputs := map[string]Input{
		"custom_input": {
			Type:        "string",
			Description: "Custom input",
			Default:     "default",
			Examples:    []any{"example"},
			Optional:    false,
		},
	}

	allInputs := appendCommonInputs(customInputs)

	t.Run("includes common inputs", func(t *testing.T) {
		// Check task input
		taskInput, ok := allInputs[inputTask]
		require.True(t, ok, "task input should be present")
		assert.Equal(t, "string", taskInput.Type)
		assert.Equal(t, "Task (without parameters) the user wants to complete with the tool.", taskInput.Description)
		assert.False(t, taskInput.Optional)

		// Check working directory input
		wdInput, ok := allInputs[inputWorkingDirectory]
		require.True(t, ok, "working directory input should be present")
		assert.Equal(t, "string", wdInput.Type)
		assert.Equal(t, "Working directory to use for the tool.", wdInput.Description)
		assert.Equal(t, ".", wdInput.Default)
		assert.True(t, wdInput.Optional)
	})

	t.Run("preserves custom inputs", func(t *testing.T) {
		customInput, ok := allInputs["custom_input"]
		require.True(t, ok, "custom input should be present")
		assert.Equal(t, "string", customInput.Type)
		assert.Equal(t, "Custom input", customInput.Description)
		assert.Equal(t, "default", customInput.Default)
		assert.Equal(t, []any{"example"}, customInput.Examples)
		assert.False(t, customInput.Optional)
	})
}

// TestValidateToolDefinition tests the validateToolDefinition function.
func TestValidateToolDefinition(t *testing.T) {
	t.Run("validates valid tool definition", func(t *testing.T) {
		def := &Definition{
			DisplayName: "Valid Tool",
			Description: "Valid Description",
			Inputs: map[string]Input{
				"input1": {
					Type:        "string",
					Description: "Valid Input",
					Default:     "default",
					Examples:    []any{"example"},
					Optional:    false,
				},
			},
		}
		err := ValidateDefinition(def)
		assert.NoError(t, err)
	})

	t.Run("validates tool definition with executable", func(t *testing.T) {
		def := &Definition{
			DisplayName: "Valid Tool",
			Description: "Valid Description",
			Executable:  "ls", // Common executable that should exist
			Inputs:      map[string]Input{},
		}
		err := ValidateDefinition(def)
		assert.NoError(t, err)
	})

	t.Run("validates tool definition with non-existent executable", func(t *testing.T) {
		def := &Definition{
			DisplayName: "Invalid Tool",
			Description: "Invalid Description",
			Executable:  "non-existent-executable",
			Inputs:      map[string]Input{},
		}
		err := ValidateDefinition(def)
		assert.ErrorContains(t, err, ErrToolExecutableNotFound)
	})

	t.Run("validates empty tool definition", func(t *testing.T) {
		def := &Definition{}
		err := ValidateDefinition(def)
		assert.ErrorContains(t, err, ErrToolMissingDisplayName)

		def.DisplayName = "Tool"
		err = ValidateDefinition(def)
		assert.ErrorContains(t, err, ErrToolMissingDescription)
	})

	t.Run("validates input fields", func(t *testing.T) {
		def := &Definition{
			DisplayName: "Tool",
			Description: "Description",
			Inputs: map[string]Input{
				"input1": {},
			},
		}
		err := ValidateDefinition(def)
		assert.ErrorContains(t, err, fmt.Sprintf("%s: %q", ErrToolInputMissingType, "input1"))

		def.Inputs["input1"] = Input{Type: "string"}
		err = ValidateDefinition(def)
		assert.ErrorContains(t, err, fmt.Sprintf("%s: %q", ErrToolInputMissingDescription, "input1"))

		def.Inputs["input1"] = Input{
			Type:        "string",
			Description: "Description",
		}
		err = ValidateDefinition(def)
		assert.NoError(t, err)
	})

	t.Run("allows empty inputs", func(t *testing.T) {
		def := &Definition{
			DisplayName: "Tool",
			Description: "Description",
		}
		err := ValidateDefinition(def)
		assert.NoError(t, err)
	})
}

// TestToolInterfaceCompliance tests that tool implementations comply with the Tool interface.
func TestToolInterfaceCompliance(t *testing.T) {
	// Test regular tool
	var _ Tool = &tool{}

	// Test exec tool
	var _ Tool = (*execTool)(nil)

	// Create and test a concrete tool instance
	logger := slog.New(slog.DiscardHandler)
	cfg := &config.ToolsConfiguration{
		Timeout: 120,
		Exec: config.ExecToolConfiguration{
			Timeout: 60,
			Shell:   "/bin/bash",
		},
	}
	def := Definition{
		DisplayName: "Test Tool",
		Description: "Test Description",
		Inputs: map[string]Input{
			"test": {
				Type:        "string",
				Description: "Test Input",
			},
		},
	}

	runner := newMockRunner(nil, nil)
	tool := New("test", def, logger, cfg, runner)

	t.Run("implements all interface methods", func(t *testing.T) {
		assert.NotPanics(t, func() {
			_ = tool.GetName()
			_ = tool.GetDisplayName()
			_ = tool.GetDescription()
			_ = tool.GetInputSchema()
			_, _ = tool.Execute(nil, context.Background())
		})
	})

	t.Run("exec tool implements all interface methods", func(t *testing.T) {
		execTool := NewExecTool(logger, cfg)
		assert.NotPanics(t, func() {
			_ = execTool.GetName()
			_ = execTool.GetDisplayName()
			_ = execTool.GetDescription()
			_ = execTool.GetInputSchema()
			_, _ = execTool.Execute(nil, context.Background())
		})
	})
}
