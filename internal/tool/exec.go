package tool

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/alextheberge/opsy/internal/config"
	"github.com/invopop/jsonschema"
)

// ExecTool is the tool for executing commands.
type execTool tool

// ExecToolName is the name of the exec tool.
const ExecToolName = "exec"

// Command is the command that was executed.
type Command struct {
	// Command is the command that was executed.
	Command string
	// WorkingDirectory is the working directory of the command.
	WorkingDirectory string
	// ExitCode is the exit code of the command.
	ExitCode int
	// Output is the output of the command.
	Output string
	// StartedAt is the time the command started.
	StartedAt time.Time
	// CompletedAt is the time the command completed.
	CompletedAt time.Time
}

const (
	// inputCommand is the input parameter for the command to execute.
	inputCommand = "command"
)

// NewExecTool creates a new exec tool.
func NewExecTool(logger *slog.Logger, cfg *config.ToolsConfiguration) *execTool {
	definition := Definition{
		DisplayName: "Exec",
		Description: fmt.Sprintf("Executes the provided shell command via the `%s` shell.", cfg.Exec.Shell),
		Inputs: map[string]Input{
			inputCommand: {
				Description: "The shell command, including all the arguments, to execute",
				Type:        "string",
				Examples: []any{
					"ls -l | grep 'myfile'",
					"git status",
					"curl -X GET https://api.example.com/data",
				},
			},
			inputWorkingDirectory: {
				Description: "The working directory for the command",
				Type:        "string",
				Examples: []any{
					"/path/to/working/directory",
					".",
				},
			},
		},
	}

	return (*execTool)(New(ExecToolName, definition, logger, cfg, nil))
}

// GetName returns the name of the tool.
func (t *execTool) GetName() string {
	return (*tool)(t).GetName()
}

// GetDisplayName returns the display name of the tool.
func (t *execTool) GetDisplayName() string {
	return (*tool)(t).GetDisplayName()
}

// GetDescription returns the description of the tool.
func (t *execTool) GetDescription() string {
	return (*tool)(t).GetDescription()
}

// GetInputSchema returns the input schema of the tool.
func (t *execTool) GetInputSchema() *jsonschema.Schema {
	return (*tool)(t).GetInputSchema()
}

// Execute executes the tool.
func (t *execTool) Execute(inputs map[string]any, ctx context.Context) (*Output, error) {
	command, ok := inputs[inputCommand].(string)
	if !ok {
		return nil, fmt.Errorf("%s: %s", ErrInvalidToolInputType, inputCommand)
	}

	workingDirectory := getWorkingDirectory(inputs)
	ctx, cancel := context.WithTimeout(ctx, t.getTimeout())
	defer cancel()

	cmd := exec.CommandContext(ctx, t.config.Exec.Shell, "-c", command)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Dir = workingDirectory
	cmd.Stdin = nil
	startedAt := time.Now()

	logger := t.logger.With("command", cmd.String()).With("working_directory", workingDirectory)
	logger.Debug("Executing command.")

	toolOutput, err := cmd.CombinedOutput()
	output := &Output{
		Tool:    t.GetName(),
		Result:  strings.TrimSpace(string(toolOutput)),
		IsError: false,
		ExecutedCommand: &Command{
			Command:          command,
			WorkingDirectory: workingDirectory,
			ExitCode:         cmd.ProcessState.ExitCode(),
			StartedAt:        startedAt,
			CompletedAt:      time.Now(),
		},
	}

	if toolOutput != nil {
		output.ExecutedCommand.Output = output.Result
	}

	if err != nil {
		logger.With("error", err).With("exit_code", cmd.ProcessState.ExitCode()).Error("Command execution failed.")
		output.IsError = true
	}

	return output, err
}

// getTimeout returns the timeout for the Exec tool.
func (t *execTool) getTimeout() time.Duration {
	timeout := t.config.Timeout
	if t.config.Exec.Timeout > 0 {
		timeout = t.config.Exec.Timeout
	}

	return time.Duration(timeout) * time.Second
}

// getWorkingDirectory returns the working directory for the Exec tool.
func getWorkingDirectory(inputs map[string]any) string {
	currentDir, _ := os.Getwd()
	currentDir = strings.TrimRight(currentDir, string(os.PathSeparator))

	workingDir, ok := inputs[inputWorkingDirectory].(string)
	if !ok || workingDir == "." {
		return currentDir
	}

	// Handle paths starting with ./
	if strings.HasPrefix(workingDir, "."+string(os.PathSeparator)) {
		return strings.TrimRight(filepath.Join(currentDir, strings.TrimPrefix(workingDir, "."+string(os.PathSeparator))), string(os.PathSeparator))
	}

	// Handle absolute paths
	if strings.HasPrefix(workingDir, string(os.PathSeparator)) {
		return strings.TrimRight(workingDir, string(os.PathSeparator))
	}

	// Handle relative paths (without ./ prefix)
	return strings.TrimRight(filepath.Join(currentDir, workingDir), string(os.PathSeparator))
}
