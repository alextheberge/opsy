package messagespane

import (
	"fmt"
	"regexp"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/alextheberge/opsy/internal/agent"
	"github.com/alextheberge/opsy/internal/thememanager"
	"github.com/stretchr/testify/assert"
)

// stripANSI removes ANSI color codes from a string.
func stripANSI(str string) string {
	re := regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]`)
	return re.ReplaceAllString(str, "")
}

// TestNew tests the creation of a new messages pane component.
func TestNew(t *testing.T) {
	theme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#000000",
			Base02: "#111111",
			Base03: "#222222",
			Base04: "#333333",
		},
		AccentColors: thememanager.AccentColors{
			Accent1: "#FF0000",
			Accent2: "#00FF00",
		},
	}

	m := New(
		WithTheme(theme),
	)

	assert.NotNil(t, m)
	assert.Equal(t, theme, m.theme)
	assert.NotNil(t, m.viewport)
	assert.Empty(t, m.messages)
}

// TestUpdate tests the update function of the messages pane component.
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

	// Test message handling
	testMsg := agent.Message{
		Message:   "Test message",
		Tool:      "",
		Timestamp: time.Now(),
	}
	m, cmd = m.Update(testMsg)
	assert.Nil(t, cmd)
	assert.Len(t, m.messages, 1)
	assert.Equal(t, testMsg, m.messages[0])
}

// TestView tests the view function of the messages pane component.
func TestView(t *testing.T) {
	theme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#000000",
			Base02: "#111111",
			Base03: "#222222",
			Base04: "#333333",
		},
		AccentColors: thememanager.AccentColors{
			Accent1: "#FF0000",
			Accent2: "#00FF00",
		},
	}

	m := New(
		WithTheme(theme),
	)

	// Set dimensions to test rendering
	m, _ = m.Update(tea.WindowSizeMsg{Width: 100, Height: 50})

	// Test initial view (empty messages)
	view := stripANSI(m.View())
	assert.NotEmpty(t, view)
	assert.Contains(t, view, "Messages")

	// Add test messages
	now := time.Now()
	m.Update(agent.Message{
		Message:   "Hello",
		Tool:      "",
		Timestamp: now,
	})
	m.Update(agent.Message{
		Message:   "Running git command",
		Tool:      "Git",
		Timestamp: now,
	})

	// Test view with messages
	view = stripANSI(m.View())
	assert.Contains(t, view, "Messages")
	assert.Contains(t, view, "Opsy:")
	assert.Contains(t, view, "Opsy->Git:")
	assert.Contains(t, view, "Hello")
	assert.Contains(t, view, "Running git command")
}

// TestInit tests the initialization of the messages pane component.
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

// TestMessageSanitization tests the message sanitization functionality.
func TestMessageSanitization(t *testing.T) {
	theme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#000000",
		},
	}
	m := New(WithTheme(theme))
	m.Update(tea.WindowSizeMsg{Width: 100, Height: 50})

	testCases := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "removes XML tags",
			input:    "<tag>content</tag>",
			expected: "content",
		},
		{
			name:     "trims whitespace",
			input:    "  message  \n\n",
			expected: "message",
		},
		{
			name:     "handles multiple tags",
			input:    "<tag1>content1</tag1><tag2>content2</tag2>",
			expected: "content1content2",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m.Update(agent.Message{
				Message:   tc.input,
				Timestamp: time.Now(),
			})
			view := stripANSI(m.View())
			assert.Contains(t, view, tc.expected)
			assert.NotContains(t, view, "<tag>")
		})
	}
}

// TestLongMessageWrapping tests the wrapping of long messages.
func TestLongMessageWrapping(t *testing.T) {
	theme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#000000",
		},
	}
	m := New(WithTheme(theme))

	// Set a narrow width to force wrapping
	m.Update(tea.WindowSizeMsg{Width: 40, Height: 50})

	longMessage := "This is a very long message that should be wrapped to multiple lines when the width is limited"
	m.Update(agent.Message{
		Message:   longMessage,
		Timestamp: time.Now(),
	})

	view := stripANSI(m.View())
	lines := regexp.MustCompile(`\n`).Split(view, -1)

	// Count lines containing parts of the message
	messageLines := 0
	for _, line := range lines {
		if strings.Contains(line, "This") || strings.Contains(line, "long") || strings.Contains(line, "limited") {
			messageLines++
		}
	}

	assert.Greater(t, messageLines, 1, "long message should be wrapped to multiple lines")
}

// TestThemeChange tests the component's response to theme changes.
func TestThemeChange(t *testing.T) {
	initialTheme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#000000",
			Base04: "#FFFFFF",
		},
		AccentColors: thememanager.AccentColors{
			Accent1: "#FF0000",
			Accent2: "#00FF00",
		},
	}

	newTheme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#111111",
			Base04: "#EEEEEE",
		},
		AccentColors: thememanager.AccentColors{
			Accent1: "#FF1111",
			Accent2: "#11FF11",
		},
	}

	// Create and setup first model
	m1 := New(WithTheme(initialTheme))
	m1.Update(tea.WindowSizeMsg{Width: 80, Height: 40})
	m1.Update(agent.Message{
		Message:   "Test message",
		Timestamp: time.Now(),
	})

	// Create and setup second model
	m2 := New(WithTheme(newTheme))
	m2.Update(tea.WindowSizeMsg{Width: 80, Height: 40})
	m2.Update(agent.Message{
		Message:   "Test message",
		Timestamp: time.Now(),
	})

	// Verify container styles are different
	assert.NotEqual(t,
		m1.containerStyle().GetBackground(),
		m2.containerStyle().GetBackground(),
		"container styles should have different backgrounds",
	)

	// Verify message styles are different
	assert.NotEqual(t,
		m1.messageStyle().GetForeground(),
		m2.messageStyle().GetForeground(),
		"message styles should have different colors",
	)

	// Verify styles match their themes
	assert.Equal(t,
		lipgloss.Color(initialTheme.BaseColors.Base01),
		m1.containerStyle().GetBackground(),
		"container style should use Base01 color",
	)

	assert.Equal(t,
		lipgloss.Color(initialTheme.BaseColors.Base04),
		m1.messageStyle().GetForeground(),
		"message style should use Base04 color",
	)

	// Verify content is identical
	stripped1 := stripANSI(m1.View())
	stripped2 := stripANSI(m2.View())
	assert.Equal(t, stripped1, stripped2, "content should be same after stripping ANSI codes")
}

// TestConcurrentAccess tests message handling with multiple updates.
func TestConcurrentAccess(t *testing.T) {
	theme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#000000",
			Base02: "#111111",
			Base03: "#222222",
			Base04: "#333333",
		},
		AccentColors: thememanager.AccentColors{
			Accent1: "#FF0000",
			Accent2: "#00FF00",
		},
	}

	m := New(WithTheme(theme))

	// Initialize viewport with window size
	m, _ = m.Update(tea.WindowSizeMsg{Width: 100, Height: 50})

	// Add messages sequentially with fixed timestamp
	timestamp := time.Date(2024, 1, 1, 10, 43, 56, 0, time.UTC)
	for i := 0; i < 10; i++ {
		msg := agent.Message{
			Message:   fmt.Sprintf("Message %d", i),
			Timestamp: timestamp,
		}
		m, _ = m.Update(msg)
	}

	// Verify that all messages are in the model's messages slice
	assert.Equal(t, 10, len(m.messages), "should have 10 messages")
	for i := 0; i < 10; i++ {
		expectedMessage := fmt.Sprintf("Message %d", i)
		assert.Equal(t, expectedMessage, m.messages[i].Message, "message %d should match", i)
	}

	// Verify that the viewport content is not empty
	content := stripANSI(m.viewport.View())
	assert.NotEmpty(t, content, "viewport content should not be empty")
}
