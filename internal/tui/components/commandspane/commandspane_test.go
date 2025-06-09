package commandspane

import (
	"regexp"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/alextheberge/opsy/internal/thememanager"
	"github.com/alextheberge/opsy/internal/tool"
	"github.com/stretchr/testify/assert"
)

// stripANSI removes ANSI color codes from a string.
func stripANSI(str string) string {
	re := regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]`)
	return re.ReplaceAllString(str, "")
}

// TestNew tests the creation of a new commands pane component.
func TestNew(t *testing.T) {
	theme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#000000",
			Base02: "#111111",
			Base03: "#222222",
			Base04: "#333333",
		},
		AccentColors: thememanager.AccentColors{
			Accent0: "#FF0000",
		},
	}

	m := New(
		WithTheme(theme),
	)

	assert.NotNil(t, m)
	assert.Equal(t, theme, m.theme)
	assert.NotNil(t, m.viewport)
	assert.Empty(t, m.commands)
}

// TestUpdate tests the update function of the commands pane component.
func TestUpdate(t *testing.T) {
	theme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#000000",
			Base02: "#111111",
			Base03: "#222222",
			Base04: "#333333",
		},
	}
	m := New(WithTheme(theme))

	// Test window size message
	newModel, cmd := m.Update(tea.WindowSizeMsg{Width: 100, Height: 50})
	assert.NotNil(t, newModel)
	assert.Nil(t, cmd)
	assert.Equal(t, 94, newModel.maxWidth) // Width - 6 for padding
	assert.Equal(t, 50, newModel.maxHeight)
	assert.Equal(t, 94, newModel.viewport.Width)
	assert.Equal(t, 50, newModel.viewport.Height)

	// Test command message
	now := time.Now()
	testCmd := tool.Command{
		Command:          "ls -la",
		WorkingDirectory: "~/opsy",
		StartedAt:        now,
	}
	m, cmd = m.Update(testCmd)
	assert.Nil(t, cmd)
	assert.Len(t, m.commands, 1)
	assert.Equal(t, testCmd, m.commands[0])
}

// TestView tests the view function of the commands pane component.
func TestView(t *testing.T) {
	theme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#000000",
			Base02: "#111111",
			Base03: "#222222",
			Base04: "#333333",
		},
		AccentColors: thememanager.AccentColors{
			Accent0: "#FF0000",
		},
	}

	m := New(
		WithTheme(theme),
	)

	// Set dimensions to test rendering
	m, _ = m.Update(tea.WindowSizeMsg{Width: 100, Height: 50})

	// Test initial view (empty commands)
	view := stripANSI(m.View())
	assert.NotEmpty(t, view)
	assert.Contains(t, view, "Commands")

	// Add test command
	now := time.Now()
	m.Update(tool.Command{
		Command:          "ls -la",
		WorkingDirectory: "~/opsy",
		StartedAt:        now,
	})

	// Test view with command
	view = stripANSI(m.View())
	assert.Contains(t, view, "Commands")
	assert.Contains(t, view, "~/opsy")
	assert.Contains(t, view, "ls -la")
	assert.Contains(t, view, now.Format("15:04:05"))
}

// TestInit tests the initialization of the commands pane component.
func TestInit(t *testing.T) {
	theme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#000000",
		},
	}
	m := New(WithTheme(theme))
	cmd := m.Init()
	assert.Nil(t, cmd)
}

// TestCommandWrapping tests the wrapping behavior of long commands.
func TestCommandWrapping(t *testing.T) {
	m := New()
	m, _ = m.Update(tea.WindowSizeMsg{Width: 40, Height: 40})

	// Create a command that will definitely wrap
	longCommand := "ls -la /very/long/path/that/will/definitely/wrap/across/multiple/lines/in/the/terminal/output/when/rendered"
	cmd := tool.Command{
		Command:          longCommand,
		WorkingDirectory: "~/opsy",
		StartedAt:        time.Now(),
	}
	m, _ = m.Update(cmd)

	// Get the view
	view := m.View()

	// Count the number of lines in the view
	lines := strings.Split(view, "\n")
	nonEmptyLines := 0
	for _, line := range lines {
		if strings.TrimSpace(line) != "" {
			nonEmptyLines++
		}
	}

	// With a width of 40, the command should wrap to at least 3 lines:
	// 1. Line with timestamp and start of command
	// 2. At least one wrapped line
	// 3. Line with working directory
	assert.GreaterOrEqual(t, nonEmptyLines, 3, "command should wrap to at least 3 lines")

	// Strip ANSI codes for easier testing
	plainView := stripANSI(view)

	// Verify the command is present and wrapped
	assert.Contains(t, plainView, "ls -la /very/l")
	assert.Contains(t, plainView, "ong/path/that/")
	assert.Contains(t, plainView, "will/definitel")
	assert.Contains(t, plainView, "y/wrap/across/")
	assert.Contains(t, plainView, "multiple/lines")
	assert.Contains(t, plainView, "/in/the/termin")
	assert.Contains(t, plainView, "al/output/when")
	assert.Contains(t, plainView, "/rendered")

	// Verify working directory is present
	assert.Contains(t, plainView, cmd.WorkingDirectory)
}

// TestMultipleCommands tests rendering of multiple commands.
func TestMultipleCommands(t *testing.T) {
	theme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#000000",
			Base02: "#111111",
			Base03: "#222222",
			Base04: "#333333",
		},
		AccentColors: thememanager.AccentColors{
			Accent0: "#FF0000",
		},
	}

	m := New(WithTheme(theme))
	m, _ = m.Update(tea.WindowSizeMsg{Width: 100, Height: 50})

	// Add multiple commands
	commands := []tool.Command{
		{
			Command:          "git status",
			WorkingDirectory: "~/project1",
			StartedAt:        time.Now(),
		},
		{
			Command:          "make build",
			WorkingDirectory: "~/project2",
			StartedAt:        time.Now().Add(time.Second),
		},
		{
			Command:          "docker ps",
			WorkingDirectory: "~/project3",
			StartedAt:        time.Now().Add(2 * time.Second),
		},
	}

	for _, cmd := range commands {
		m, _ = m.Update(cmd)
	}

	view := stripANSI(m.View())

	// Verify all commands are rendered
	for _, cmd := range commands {
		assert.Contains(t, view, cmd.Command)
		assert.Contains(t, view, cmd.WorkingDirectory)
		assert.Contains(t, view, cmd.StartedAt.Format("15:04:05"))
	}

	// Verify order (last command should be at the bottom)
	lastCmdIndex := strings.LastIndex(view, commands[len(commands)-1].Command)
	for i := 0; i < len(commands)-1; i++ {
		cmdIndex := strings.LastIndex(view, commands[i].Command)
		assert.Less(t, cmdIndex, lastCmdIndex, "commands should be in chronological order")
	}
}

// TestThemeChange tests the component's response to theme changes.
func TestThemeChange(t *testing.T) {
	initialTheme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base00: "#000000",
			Base01: "#111111",
			Base02: "#222222",
			Base03: "#333333",
			Base04: "#444444",
		},
		AccentColors: thememanager.AccentColors{
			Accent0: "#FF0000",
			Accent1: "#00FF00",
			Accent2: "#0000FF",
		},
	}

	newTheme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base00: "#FFFFFF",
			Base01: "#EEEEEE",
			Base02: "#DDDDDD",
			Base03: "#CCCCCC",
			Base04: "#BBBBBB",
		},
		AccentColors: thememanager.AccentColors{
			Accent0: "#00FF00",
			Accent1: "#FF0000",
			Accent2: "#0000FF",
		},
	}

	// Create models with different themes
	m1 := New(WithTheme(initialTheme))
	m2 := New(WithTheme(newTheme))

	// Verify that styles are different
	assert.NotEqual(t,
		m1.commandStyle().GetForeground(),
		m2.commandStyle().GetForeground(),
		"command styles should have different colors",
	)

	assert.NotEqual(t,
		m1.containerStyle().GetBackground(),
		m2.containerStyle().GetBackground(),
		"container styles should have different backgrounds",
	)

	assert.NotEqual(t,
		m1.workdirStyle().GetBackground(),
		m2.workdirStyle().GetBackground(),
		"workdir styles should have different backgrounds",
	)

	// Verify that the styles use the correct theme colors
	assert.Equal(t,
		lipgloss.Color(initialTheme.AccentColors.Accent0),
		m1.commandStyle().GetForeground(),
		"command style should use Accent0 color",
	)

	assert.Equal(t,
		lipgloss.Color(initialTheme.BaseColors.Base01),
		m1.containerStyle().GetBackground(),
		"container style should use Base01 color",
	)

	assert.Equal(t,
		lipgloss.Color(newTheme.AccentColors.Accent0),
		m2.commandStyle().GetForeground(),
		"command style should use Accent0 color",
	)

	assert.Equal(t,
		lipgloss.Color(newTheme.BaseColors.Base01),
		m2.containerStyle().GetBackground(),
		"container style should use Base01 color",
	)
}
