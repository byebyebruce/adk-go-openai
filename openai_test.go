package openai

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/sashabaranov/go-openai"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestConvertRoleToOpenAI(t *testing.T) {
	tests := []struct {
		name string
		role string
		want string
	}{
		{
			name: "user role",
			role: "user",
			want: openai.ChatMessageRoleUser,
		},
		{
			name: "model role",
			role: "model",
			want: openai.ChatMessageRoleAssistant,
		},
		{
			name: "system role",
			role: "system",
			want: openai.ChatMessageRoleSystem,
		},
		{
			name: "unknown role defaults to user",
			role: "unknown",
			want: openai.ChatMessageRoleUser,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertRoleToOpenAI(tt.role)
			if got != tt.want {
				t.Errorf("convertRoleToOpenAI(%q) = %q, want %q", tt.role, got, tt.want)
			}
		})
	}
}

func TestConvertFinishReason(t *testing.T) {
	tests := []struct {
		name   string
		reason string
		want   genai.FinishReason
	}{
		{
			name:   "stop",
			reason: "stop",
			want:   genai.FinishReasonStop,
		},
		{
			name:   "length",
			reason: "length",
			want:   genai.FinishReasonMaxTokens,
		},
		{
			name:   "tool_calls",
			reason: "tool_calls",
			want:   genai.FinishReasonStop,
		},
		{
			name:   "content_filter",
			reason: "content_filter",
			want:   genai.FinishReasonSafety,
		},
		{
			name:   "unknown",
			reason: "unknown",
			want:   genai.FinishReasonUnspecified,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertFinishReason(tt.reason)
			if got != tt.want {
				t.Errorf("convertFinishReason(%q) = %q, want %q", tt.reason, got, tt.want)
			}
		})
	}
}

func TestToOpenAIChatCompletionMessage(t *testing.T) {
	tests := []struct {
		name    string
		content *genai.Content
		want    openai.ChatCompletionMessage
		wantErr bool
	}{
		{
			name: "simple text message",
			content: &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: "Hello, world!"}},
			},
			want: openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello, world!",
			},
			wantErr: false,
		},
		{
			name: "message with function call",
			content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "call_123",
							Name: "get_weather",
							Args: map[string]any{"location": "Paris"},
						},
					},
				},
			},
			want: openai.ChatCompletionMessage{
				Role: openai.ChatMessageRoleAssistant,
				ToolCalls: []openai.ToolCall{
					{
						ID:   "call_123",
						Type: openai.ToolTypeFunction,
						Function: openai.FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location":"Paris"}`,
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "message with function response",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:   "call_123",
							Name: "get_weather",
							Response: map[string]any{
								"temperature": 20,
								"condition":   "sunny",
							},
						},
					},
				},
			},
			want: openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    `{"condition":"sunny","temperature":20}`,
				ToolCallID: "call_123",
			},
			wantErr: false,
		},
		{
			name: "message with inline data (image)",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "What's in this image?"},
					{
						InlineData: &genai.Blob{
							MIMEType: "image/jpeg",
							Data:     []byte("fake_image_data"),
						},
					},
				},
			},
			want: openai.ChatCompletionMessage{
				Role: openai.ChatMessageRoleUser,
				MultiContent: []openai.ChatMessagePart{
					{
						Type: openai.ChatMessagePartTypeText,
						Text: "What's in this image?",
					},
					{
						Type: openai.ChatMessagePartTypeImageURL,
						ImageURL: &openai.ChatMessageImageURL{
							URL:    "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
							Detail: openai.ImageURLDetailAuto,
						},
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := toOpenAIChatCompletionMessage(tt.content)
			if (err != nil) != tt.wantErr {
				t.Errorf("toOpenAIChatCompletionMessage() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// For function call messages, we need to compare the arguments as JSON
			if len(tt.want.ToolCalls) > 0 && len(got.ToolCalls) > 0 {
				if diff := cmp.Diff(tt.want.ToolCalls[0].ID, got.ToolCalls[0].ID); diff != "" {
					t.Errorf("ToolCall ID mismatch (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(tt.want.ToolCalls[0].Function.Name, got.ToolCalls[0].Function.Name); diff != "" {
					t.Errorf("Function Name mismatch (-want +got):\n%s", diff)
				}
				// Compare JSON arguments
				var wantArgs, gotArgs map[string]any
				json.Unmarshal([]byte(tt.want.ToolCalls[0].Function.Arguments), &wantArgs)
				json.Unmarshal([]byte(got.ToolCalls[0].Function.Arguments), &gotArgs)
				if diff := cmp.Diff(wantArgs, gotArgs); diff != "" {
					t.Errorf("Function Arguments mismatch (-want +got):\n%s", diff)
				}
				return
			}

			// For function response messages, compare as JSON
			if tt.want.Role == openai.ChatMessageRoleTool {
				var wantContent, gotContent map[string]any
				json.Unmarshal([]byte(tt.want.Content), &wantContent)
				json.Unmarshal([]byte(got.Content), &gotContent)
				if diff := cmp.Diff(wantContent, gotContent); diff != "" {
					t.Errorf("Content mismatch (-want +got):\n%s", diff)
				}
				if got.ToolCallID != tt.want.ToolCallID {
					t.Errorf("ToolCallID = %v, want %v", got.ToolCallID, tt.want.ToolCallID)
				}
				return
			}

			// For regular messages
			opts := []cmp.Option{
				cmpopts.IgnoreUnexported(openai.ChatCompletionMessage{}),
			}
			if diff := cmp.Diff(tt.want, got, opts...); diff != "" {
				t.Errorf("toOpenAIChatCompletionMessage() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConvertChatCompletionResponse(t *testing.T) {
	tests := []struct {
		name    string
		resp    *openai.ChatCompletionResponse
		want    *model.LLMResponse
		wantErr bool
	}{
		{
			name: "simple text response",
			resp: &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "Hello! How can I help you?",
						},
						FinishReason: "stop",
					},
				},
				Usage: openai.Usage{
					PromptTokens:     10,
					CompletionTokens: 8,
					TotalTokens:      18,
				},
			},
			want: &model.LLMResponse{
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{Text: "Hello! How can I help you?"}},
				},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     10,
					CandidatesTokenCount: 8,
					TotalTokenCount:      18,
				},
				FinishReason: genai.FinishReasonStop,
				TurnComplete: true,
			},
			wantErr: false,
		},
		{
			name: "response with tool call",
			resp: &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_abc123",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "get_current_weather",
										Arguments: `{"location":"San Francisco","unit":"celsius"}`,
									},
								},
							},
						},
						FinishReason: "tool_calls",
					},
				},
				Usage: openai.Usage{
					PromptTokens:     15,
					CompletionTokens: 20,
					TotalTokens:      35,
				},
			},
			want: &model.LLMResponse{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								ID:   "call_abc123",
								Name: "get_current_weather",
								Args: map[string]any{
									"location": "San Francisco",
									"unit":     "celsius",
								},
							},
						},
					},
				},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     15,
					CandidatesTokenCount: 20,
					TotalTokenCount:      35,
				},
				FinishReason: genai.FinishReasonStop,
				TurnComplete: true,
			},
			wantErr: false,
		},
		{
			name: "empty choices error",
			resp: &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{},
			},
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := convertChatCompletionResponse(tt.resp)
			if (err != nil) != tt.wantErr {
				t.Errorf("convertChatCompletionResponse() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			opts := []cmp.Option{
				cmpopts.IgnoreUnexported(genai.Content{}),
				cmpopts.IgnoreUnexported(genai.Part{}),
			}
			if diff := cmp.Diff(tt.want, got, opts...); diff != "" {
				t.Errorf("convertChatCompletionResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestToOpenAIChatCompletionRequest(t *testing.T) {
	temp := float32(0.7)
	topP := float32(0.9)

	tests := []struct {
		name      string
		req       *model.LLMRequest
		modelName string
		want      openai.ChatCompletionRequest
		wantErr   bool
	}{
		{
			name: "basic request",
			req: &model.LLMRequest{
				Contents: []*genai.Content{
					{
						Role:  "user",
						Parts: []*genai.Part{{Text: "Hello"}},
					},
				},
				Config: &genai.GenerateContentConfig{
					Temperature: &temp,
				},
			},
			modelName: "gpt-4",
			want: openai.ChatCompletionRequest{
				Model: "gpt-4",
				Messages: []openai.ChatCompletionMessage{
					{
						Role:    openai.ChatMessageRoleUser,
						Content: "Hello",
					},
				},
				Temperature: 0.7,
			},
			wantErr: false,
		},
		{
			name: "request with tools",
			req: &model.LLMRequest{
				Contents: []*genai.Content{
					{
						Role:  "user",
						Parts: []*genai.Part{{Text: "What's the weather?"}},
					},
				},
				Config: &genai.GenerateContentConfig{
					Tools: []*genai.Tool{
						{
							FunctionDeclarations: []*genai.FunctionDeclaration{
								{
									Name:        "get_weather",
									Description: "Get current weather",
									Parameters: &genai.Schema{
										Type: genai.TypeObject,
										Properties: map[string]*genai.Schema{
											"location": {
												Type:        genai.TypeString,
												Description: "City name",
											},
										},
										Required: []string{"location"},
									},
								},
							},
						},
					},
				},
			},
			modelName: "gpt-4",
			want: openai.ChatCompletionRequest{
				Model: "gpt-4",
				Messages: []openai.ChatCompletionMessage{
					{
						Role:    openai.ChatMessageRoleUser,
						Content: "What's the weather?",
					},
				},
				Tools: []openai.Tool{
					{
						Type: openai.ToolTypeFunction,
						Function: &openai.FunctionDefinition{
							Name:        "get_weather",
							Description: "Get current weather",
							Parameters: map[string]any{
								"type": "object",
								"properties": map[string]any{
									"location": map[string]any{
										"type":        "string",
										"description": "City name",
									},
								},
								"required": []string{"location"},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "request with system instruction",
			req: &model.LLMRequest{
				Contents: []*genai.Content{
					{
						Role:  "user",
						Parts: []*genai.Part{{Text: "What's the weather?"}},
					},
				},
				Config: &genai.GenerateContentConfig{
					SystemInstruction: &genai.Content{
						Parts: []*genai.Part{{Text: "You are a helpful weather assistant."}},
					},
					Temperature: &temp,
					TopP:        &topP,
				},
			},
			modelName: "gpt-4",
			want: openai.ChatCompletionRequest{
				Model: "gpt-4",
				Messages: []openai.ChatCompletionMessage{
					{
						Role:    openai.ChatMessageRoleSystem,
						Content: "You are a helpful weather assistant.",
					},
					{
						Role:    openai.ChatMessageRoleUser,
						Content: "What's the weather?",
					},
				},
				Temperature: 0.7,
				TopP:        0.9,
			},
			wantErr: false,
		},
		{
			name: "request with JSON mode",
			req: &model.LLMRequest{
				Contents: []*genai.Content{
					{
						Role:  "user",
						Parts: []*genai.Part{{Text: "Return user data"}},
					},
				},
				Config: &genai.GenerateContentConfig{
					ResponseMIMEType: "application/json",
				},
			},
			modelName: "gpt-4",
			want: openai.ChatCompletionRequest{
				Model: "gpt-4",
				Messages: []openai.ChatCompletionMessage{
					{
						Role:    openai.ChatMessageRoleUser,
						Content: "Return user data",
					},
				},
				ResponseFormat: &openai.ChatCompletionResponseFormat{
					Type: openai.ChatCompletionResponseFormatTypeJSONObject,
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := toOpenAIChatCompletionRequest(tt.req, tt.modelName)
			if (err != nil) != tt.wantErr {
				t.Errorf("toOpenAIChatCompletionRequest() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// Compare basic fields
			if got.Model != tt.want.Model {
				t.Errorf("Model = %v, want %v", got.Model, tt.want.Model)
			}
			if got.Temperature != tt.want.Temperature {
				t.Errorf("Temperature = %v, want %v", got.Temperature, tt.want.Temperature)
			}
			if got.TopP != tt.want.TopP {
				t.Errorf("TopP = %v, want %v", got.TopP, tt.want.TopP)
			}

			// Compare messages
			if len(got.Messages) != len(tt.want.Messages) {
				t.Errorf("Messages length = %v, want %v", len(got.Messages), len(tt.want.Messages))
			} else {
				for i := range got.Messages {
					if got.Messages[i].Role != tt.want.Messages[i].Role {
						t.Errorf("Message[%d].Role = %v, want %v", i, got.Messages[i].Role, tt.want.Messages[i].Role)
					}
					if got.Messages[i].Content != tt.want.Messages[i].Content {
						t.Errorf("Message[%d].Content = %v, want %v", i, got.Messages[i].Content, tt.want.Messages[i].Content)
					}
				}
			}

			// Compare ResponseFormat
			if tt.want.ResponseFormat != nil {
				if got.ResponseFormat == nil {
					t.Errorf("ResponseFormat is nil, want %v", tt.want.ResponseFormat)
				} else if got.ResponseFormat.Type != tt.want.ResponseFormat.Type {
					t.Errorf("ResponseFormat.Type = %v, want %v", got.ResponseFormat.Type, tt.want.ResponseFormat.Type)
				}
			}

			// Compare Tools
			if len(tt.want.Tools) > 0 {
				if len(got.Tools) != len(tt.want.Tools) {
					t.Errorf("Tools length = %v, want %v", len(got.Tools), len(tt.want.Tools))
				} else {
					for i := range got.Tools {
						if got.Tools[i].Type != tt.want.Tools[i].Type {
							t.Errorf("Tool[%d].Type = %v, want %v", i, got.Tools[i].Type, tt.want.Tools[i].Type)
						}
						if got.Tools[i].Function != nil && tt.want.Tools[i].Function != nil {
							if got.Tools[i].Function.Name != tt.want.Tools[i].Function.Name {
								t.Errorf("Tool[%d].Function.Name = %v, want %v", i, got.Tools[i].Function.Name, tt.want.Tools[i].Function.Name)
							}
							if got.Tools[i].Function.Description != tt.want.Tools[i].Function.Description {
								t.Errorf("Tool[%d].Function.Description = %v, want %v", i, got.Tools[i].Function.Description, tt.want.Tools[i].Function.Description)
							}
						}
					}
				}
			}
		})
	}
}

func TestConvertTools(t *testing.T) {
	tests := []struct {
		name      string
		genaiTool []*genai.Tool
		want      []openai.Tool
		wantErr   bool
	}{
		{
			name: "simple function declaration",
			genaiTool: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{
							Name:        "get_weather",
							Description: "Get the current weather",
							Parameters: &genai.Schema{
								Type: genai.TypeObject,
								Properties: map[string]*genai.Schema{
									"location": {
										Type:        genai.TypeString,
										Description: "The city and state",
									},
									"unit": {
										Type: genai.TypeString,
										Enum: []string{"celsius", "fahrenheit"},
									},
								},
								Required: []string{"location"},
							},
						},
					},
				},
			},
			want: []openai.Tool{
				{
					Type: openai.ToolTypeFunction,
					Function: &openai.FunctionDefinition{
						Name:        "get_weather",
						Description: "Get the current weather",
						Parameters: map[string]any{
							"type": "object",
							"properties": map[string]any{
								"location": map[string]any{
									"type":        "string",
									"description": "The city and state",
								},
								"unit": map[string]any{
									"type": "string",
									"enum": []string{"celsius", "fahrenheit"},
								},
							},
							"required": []string{"location"},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "multiple function declarations",
			genaiTool: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{
							Name:        "add",
							Description: "Add two numbers",
							Parameters: &genai.Schema{
								Type: genai.TypeObject,
								Properties: map[string]*genai.Schema{
									"a": {Type: genai.TypeNumber},
									"b": {Type: genai.TypeNumber},
								},
								Required: []string{"a", "b"},
							},
						},
						{
							Name:        "multiply",
							Description: "Multiply two numbers",
							Parameters: &genai.Schema{
								Type: genai.TypeObject,
								Properties: map[string]*genai.Schema{
									"x": {Type: genai.TypeNumber},
									"y": {Type: genai.TypeNumber},
								},
								Required: []string{"x", "y"},
							},
						},
					},
				},
			},
			want: []openai.Tool{
				{
					Type: openai.ToolTypeFunction,
					Function: &openai.FunctionDefinition{
						Name:        "add",
						Description: "Add two numbers",
						Parameters: map[string]any{
							"type": "object",
							"properties": map[string]any{
								"a": map[string]any{"type": "number"},
								"b": map[string]any{"type": "number"},
							},
							"required": []string{"a", "b"},
						},
					},
				},
				{
					Type: openai.ToolTypeFunction,
					Function: &openai.FunctionDefinition{
						Name:        "multiply",
						Description: "Multiply two numbers",
						Parameters: map[string]any{
							"type": "object",
							"properties": map[string]any{
								"x": map[string]any{"type": "number"},
								"y": map[string]any{"type": "number"},
							},
							"required": []string{"x", "y"},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name:      "empty tools",
			genaiTool: []*genai.Tool{},
			want:      []openai.Tool{},
			wantErr:   false,
		},
		{
			name:      "nil tools",
			genaiTool: nil,
			want:      []openai.Tool{},
			wantErr:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := convertTools(tt.genaiTool)
			if (err != nil) != tt.wantErr {
				t.Errorf("convertTools() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if len(got) != len(tt.want) {
				t.Errorf("convertTools() returned %d tools, want %d", len(got), len(tt.want))
				return
			}

			for i := range got {
				if got[i].Type != tt.want[i].Type {
					t.Errorf("Tool[%d].Type = %v, want %v", i, got[i].Type, tt.want[i].Type)
				}
				if got[i].Function == nil || tt.want[i].Function == nil {
					if got[i].Function != tt.want[i].Function {
						t.Errorf("Tool[%d].Function mismatch", i)
					}
					continue
				}
				if got[i].Function.Name != tt.want[i].Function.Name {
					t.Errorf("Tool[%d].Function.Name = %v, want %v", i, got[i].Function.Name, tt.want[i].Function.Name)
				}
				if got[i].Function.Description != tt.want[i].Function.Description {
					t.Errorf("Tool[%d].Function.Description = %v, want %v", i, got[i].Function.Description, tt.want[i].Function.Description)
				}
			}
		})
	}
}

func TestExtractTextFromContent(t *testing.T) {
	tests := []struct {
		name    string
		content *genai.Content
		want    string
	}{
		{
			name:    "nil content",
			content: nil,
			want:    "",
		},
		{
			name: "single text part",
			content: &genai.Content{
				Parts: []*genai.Part{{Text: "Hello"}},
			},
			want: "Hello",
		},
		{
			name: "multiple text parts",
			content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "Hello"},
					{Text: "World"},
				},
			},
			want: "Hello\nWorld",
		},
		{
			name: "mixed parts with non-text",
			content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "Hello"},
					{FunctionCall: &genai.FunctionCall{Name: "test"}},
					{Text: "World"},
				},
			},
			want: "Hello\nWorld",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractTextFromContent(tt.content)
			if got != tt.want {
				t.Errorf("extractTextFromContent() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestParseJSONArgs(t *testing.T) {
	tests := []struct {
		name     string
		argsJSON string
		want     map[string]any
	}{
		{
			name:     "empty string",
			argsJSON: "",
			want:     map[string]any{},
		},
		{
			name:     "valid JSON",
			argsJSON: `{"key": "value", "number": 42}`,
			want:     map[string]any{"key": "value", "number": float64(42)},
		},
		{
			name:     "invalid JSON returns empty map",
			argsJSON: `{invalid}`,
			want:     map[string]any{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseJSONArgs(tt.argsJSON)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("parseJSONArgs() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestOpenAIModel_Name(t *testing.T) {
	modelName := "gpt-4-turbo"
	model := NewOpenAIModelWithAPIKey(modelName, "xx")

	if got := model.Name(); got != modelName {
		t.Errorf("Name() = %v, want %v", got, modelName)
	}
}

// TestOpenAIModel_GenerateContent would require mocking the OpenAI client
// which is complex. In practice, this would be tested with integration tests
// or by using a mock server.
func TestOpenAIModel_GenerateContent_Interface(t *testing.T) {
	// Verify that OpenAIModel implements model.LLM interface
	var _ model.LLM = &OpenAIModel{}

	modelName := "gpt-4"
	model := NewOpenAIModel(modelName, openai.ClientConfig{})

	// Verify the model can be created and has the right type
	if model == nil {
		t.Error("NewOpenAIModel() returned nil")
	}
	if model.ModelName != "gpt-4" {
		t.Errorf("ModelName = %v, want %v", model.ModelName, "gpt-4")
	}
}

// Benchmark tests
func TestToolCallBuilder(t *testing.T) {
	// Simulate the streaming scenario where tool call comes in multiple chunks
	builder := &toolCallBuilder{
		id:   "call_123",
		name: "get_weather",
		args: "",
	}

	// First chunk: ID and name
	builder.id = "call_123"
	builder.name = "get_weather"

	// Second chunk: partial arguments
	builder.args += `{"location":`

	// Third chunk: more arguments
	builder.args += `"Paris"`

	// Fourth chunk: closing arguments
	builder.args += `}`

	// Verify the final state
	if builder.id != "call_123" {
		t.Errorf("ID = %v, want %v", builder.id, "call_123")
	}
	if builder.name != "get_weather" {
		t.Errorf("Name = %v, want %v", builder.name, "get_weather")
	}

	expectedArgs := `{"location":"Paris"}`
	if builder.args != expectedArgs {
		t.Errorf("Args = %v, want %v", builder.args, expectedArgs)
	}

	// Parse and verify the arguments
	args := parseJSONArgs(builder.args)
	if args["location"] != "Paris" {
		t.Errorf("Parsed location = %v, want Paris", args["location"])
	}
}

func BenchmarkConvertRoleToOpenAI(b *testing.B) {
	for i := 0; i < b.N; i++ {
		convertRoleToOpenAI("user")
	}
}

func BenchmarkToOpenAIChatCompletionMessage(b *testing.B) {
	content := &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{{Text: "Hello, world!"}},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = toOpenAIChatCompletionMessage(content)
	}
}

func BenchmarkConvertChatCompletionResponse(b *testing.B) {
	resp := &openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleAssistant,
					Content: "Hello! How can I help you?",
				},
				FinishReason: "stop",
			},
		},
		Usage: openai.Usage{
			PromptTokens:     10,
			CompletionTokens: 8,
			TotalTokens:      18,
		},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = convertChatCompletionResponse(resp)
	}
}
