package header

import (
	"regexp"
	"sync"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/alextheberge/opsy/internal/thememanager"
	"github.com/stretchr/testify/assert"
)

// stripANSI removes ANSI color codes from a string.
func stripANSI(str string) string {
	re := regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]`)
	return re.ReplaceAllString(str, "")
}

// TestHeaderCreation tests the creation of a new header component.
func TestHeaderCreation(t *testing.T) {
	t.Run("creates with default values", func(t *testing.T) {
		header := New()
		assert.NotNil(t, header)
		assert.Empty(t, header.task)
		assert.Equal(t, thememanager.Theme{}, header.theme)
	})

	t.Run("creates with task", func(t *testing.T) {
		task := "Test Task"
		header := New(WithTask(task))
		assert.Equal(t, task, header.task)
	})

	t.Run("creates with theme", func(t *testing.T) {
		theme := thememanager.Theme{
			BaseColors: thememanager.BaseColors{
				Base01: "#000000",
				Base04: "#FFFFFF",
			},
		}
		header := New(WithTheme(theme))
		assert.Equal(t, theme, header.theme)
	})
}

// TestHeaderUpdate tests the update function of the header component.
func TestHeaderUpdate(t *testing.T) {
	t.Run("handles window size update", func(t *testing.T) {
		header := New()
		newWidth := 100
		updatedHeader, cmd := header.Update(tea.WindowSizeMsg{Width: newWidth})
		assert.NotNil(t, updatedHeader)
		assert.Nil(t, cmd)
		assert.Equal(t, newWidth, updatedHeader.maxWidth)
	})
}

// TestHeaderView tests the view function of the header component.
func TestHeaderView(t *testing.T) {
	theme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#000000",
			Base04: "#FFFFFF",
		},
	}

	testCases := []struct {
		name     string
		task     string
		width    int
		contains []string
	}{
		{
			name:     "empty task",
			task:     "",
			width:    100,
			contains: []string{"Task:", ""},
		},
		{
			name:     "with task",
			task:     "Test Task",
			width:    100,
			contains: []string{"Task:", "Test Task"},
		},
		{
			name:     "long task with wrapping",
			task:     "This is a very long task that should be wrapped to multiple lines when the width is limited",
			width:    40,
			contains: []string{"Task:", "This is a very", "long task that"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			header := New(
				WithTask(tc.task),
				WithTheme(theme),
			)
			header.maxWidth = tc.width

			view := stripANSI(header.View())

			for _, expected := range tc.contains {
				assert.Contains(t, view, expected)
			}
		})
	}
}

// TestHeaderOptions tests the option functions of the header component.
func TestHeaderOptions(t *testing.T) {
	theme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#000000",
			Base04: "#FFFFFF",
		},
	}

	testCases := []struct {
		name          string
		options       []Option
		expectedTask  string
		expectedTheme thememanager.Theme
	}{
		{
			name:          "with task option",
			options:       []Option{WithTask("Test Task")},
			expectedTask:  "Test Task",
			expectedTheme: thememanager.Theme{},
		},
		{
			name:          "with theme option",
			options:       []Option{WithTheme(theme)},
			expectedTask:  "",
			expectedTheme: theme,
		},
		{
			name: "with both options",
			options: []Option{
				WithTask("Test Task"),
				WithTheme(theme),
			},
			expectedTask:  "Test Task",
			expectedTheme: theme,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			header := New(tc.options...)
			assert.Equal(t, tc.expectedTask, header.task)
			assert.Equal(t, tc.expectedTheme, header.theme)
		})
	}
}

// TestThemeChange tests the component's response to theme changes.
func TestThemeChange(t *testing.T) {
	initialTheme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#000000",
			Base04: "#FFFFFF",
		},
	}

	newTheme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#111111",
			Base04: "#EEEEEE",
		},
	}

	task := "Test Task"

	// Create and setup first model
	m1 := New(WithTheme(initialTheme), WithTask(task))
	m1, _ = m1.Update(tea.WindowSizeMsg{Width: 100})

	// Create and setup second model
	m2 := New(WithTheme(newTheme), WithTask(task))
	m2, _ = m2.Update(tea.WindowSizeMsg{Width: 100})

	// Verify container styles are different
	assert.NotEqual(t,
		m1.containerStyle.GetBackground(),
		m2.containerStyle.GetBackground(),
		"container styles should have different backgrounds",
	)

	// Verify text styles are different
	assert.NotEqual(t,
		m1.textStyle.GetForeground(),
		m2.textStyle.GetForeground(),
		"text styles should have different colors",
	)

	// Verify styles match their themes
	assert.Equal(t,
		lipgloss.Color(initialTheme.BaseColors.Base01),
		m1.containerStyle.GetBackground(),
		"container style should use Base01 color",
	)

	assert.Equal(t,
		lipgloss.Color(initialTheme.BaseColors.Base04),
		m1.textStyle.GetForeground(),
		"text style should use Base04 color",
	)

	// Verify content is identical
	stripped1 := stripANSI(m1.View())
	stripped2 := stripANSI(m2.View())
	assert.Equal(t, stripped1, stripped2, "content should be same after stripping ANSI codes")
}

// TestConcurrentAccess tests thread safety of the header component.
func TestConcurrentAccess(t *testing.T) {
	m := New(WithTask("Test Task"))
	var wg sync.WaitGroup
	numGoroutines := 10

	// Test concurrent updates
	wg.Add(numGoroutines)
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			_, _ = m.Update(tea.WindowSizeMsg{Width: 100})
			_ = m.View()
		}()
	}
	wg.Wait()

	// Verify component is still in a valid state
	view := stripANSI(m.View())
	assert.NotEmpty(t, view)
	assert.Contains(t, view, "Test Task")
}
