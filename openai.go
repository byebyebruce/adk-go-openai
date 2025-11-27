package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"

	"github.com/sashabaranov/go-openai"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

var _ model.LLM = &OpenAIModel{}

var (
	ErrNoChoicesInResponse   = errors.New("no choices in OpenAI response")
	ErrUnknownPartInResponse = errors.New("unknown part type in genai content")
)

type OpenAIModel struct {
	Client    *openai.Client
	ModelName string
}

func NewOpenAIModelWithAPIKey(modelName string, apiKey string) *OpenAIModel {
	cfg := openai.DefaultConfig(apiKey)
	return NewOpenAIModel(modelName, cfg)
}

func NewOpenAIModel(modelName string, cfg openai.ClientConfig) *OpenAIModel {
	client := openai.NewClientWithConfig(cfg)
	return &OpenAIModel{
		Client:    client,
		ModelName: modelName,
	}
}

// Name implements model.LLM.
func (o *OpenAIModel) Name() string {
	return o.ModelName
}

// GenerateContent implements model.LLM.
func (o *OpenAIModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if stream {
		return o.generateStream(ctx, req)
	}
	return o.generate(ctx, req)
}

func (o *OpenAIModel) generate(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		openaiReq, err := toOpenAIChatCompletionRequest(req, o.ModelName)
		if err != nil {
			yield(nil, err)
			return
		}

		resp, err := o.Client.CreateChatCompletion(ctx, openaiReq)
		if err != nil {
			yield(nil, err)
			return
		}

		llmResp, err := convertChatCompletionResponse(&resp)
		if err != nil {
			yield(nil, err)
			return
		}

		yield(llmResp, nil)
	}
}

func (o *OpenAIModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		openaiReq, err := toOpenAIChatCompletionRequest(req, o.ModelName)
		if err != nil {
			yield(nil, err)
			return
		}
		openaiReq.Stream = true

		stream, err := o.Client.CreateChatCompletionStream(ctx, openaiReq)
		if err != nil {
			yield(nil, err)
			return
		}
		defer stream.Close()

		// Aggregate the streaming chunks
		aggregatedContent := &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{},
		}
		var finishReason genai.FinishReason
		var usageMetadata *genai.GenerateContentResponseUsageMetadata

		// Track tool calls by index to properly aggregate them across chunks
		toolCallsMap := make(map[int]*toolCallBuilder)

		for {
			chunk, err := stream.Recv()
			if err != nil {
				if errors.Is(err, io.EOF) {
					break
				}
				yield(nil, err)
				return
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]

			// Handle delta content
			if choice.Delta.Content != "" {
				part := &genai.Part{Text: choice.Delta.Content}
				aggregatedContent.Parts = append(aggregatedContent.Parts, part)

				// Yield partial response
				llmResp := &model.LLMResponse{
					Content:      &genai.Content{Role: "model", Parts: []*genai.Part{part}},
					Partial:      true,
					TurnComplete: false,
				}
				if !yield(llmResp, nil) {
					return
				}
			}

			// Handle tool calls in delta - aggregate across chunks
			if len(choice.Delta.ToolCalls) > 0 {
				for _, toolCall := range choice.Delta.ToolCalls {
					// Use Index if available, otherwise use 0 as default
					idx := 0
					if toolCall.Index != nil {
						idx = *toolCall.Index
					}

					builder, exists := toolCallsMap[idx]
					if !exists {
						builder = &toolCallBuilder{
							id:   toolCall.ID,
							name: toolCall.Function.Name,
							args: "",
						}
						toolCallsMap[idx] = builder
					}

					// Update fields if present
					if toolCall.ID != "" {
						builder.id = toolCall.ID
					}
					if toolCall.Function.Name != "" {
						builder.name = toolCall.Function.Name
					}
					if toolCall.Function.Arguments != "" {
						builder.args += toolCall.Function.Arguments
					}
				}
			}

			// Capture finish reason
			if choice.FinishReason != "" {
				finishReason = convertFinishReason(string(choice.FinishReason))
			}

			// Capture usage metadata if available
			if chunk.Usage != nil {
				usageMetadata = &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     int32(chunk.Usage.PromptTokens),
					CandidatesTokenCount: int32(chunk.Usage.CompletionTokens),
					TotalTokenCount:      int32(chunk.Usage.TotalTokens),
				}
			}
		}

		// Convert aggregated tool calls to parts
		if len(toolCallsMap) > 0 {
			// Sort by index to maintain order
			indices := make([]int, 0, len(toolCallsMap))
			for idx := range toolCallsMap {
				indices = append(indices, idx)
			}
			// Simple bubble sort for small arrays
			for i := 0; i < len(indices)-1; i++ {
				for j := 0; j < len(indices)-i-1; j++ {
					if indices[j] > indices[j+1] {
						indices[j], indices[j+1] = indices[j+1], indices[j]
					}
				}
			}

			for _, idx := range indices {
				builder := toolCallsMap[idx]
				part := &genai.Part{
					FunctionCall: &genai.FunctionCall{
						ID:   builder.id,
						Name: builder.name,
						Args: parseJSONArgs(builder.args),
					},
				}
				aggregatedContent.Parts = append(aggregatedContent.Parts, part)
			}
		}

		// Send final complete response
		finalResp := &model.LLMResponse{
			Content:       aggregatedContent,
			UsageMetadata: usageMetadata,
			FinishReason:  finishReason,
			Partial:       false,
			TurnComplete:  true,
		}
		yield(finalResp, nil)
	}
}

// toolCallBuilder helps aggregate tool call information across streaming chunks
type toolCallBuilder struct {
	id   string
	name string
	args string
}

func toOpenAIChatCompletionRequest(req *model.LLMRequest, modelName string) (openai.ChatCompletionRequest, error) {
	openaiMessages := make([]openai.ChatCompletionMessage, 0, len(req.Contents))
	for _, content := range req.Contents {
		msg, err := toOpenAIChatCompletionMessage(content)
		if err != nil {
			return openai.ChatCompletionRequest{}, err
		}
		openaiMessages = append(openaiMessages, msg)
	}

	openaiReq := openai.ChatCompletionRequest{
		Model:    modelName,
		Messages: openaiMessages,
	}

	// Convert tools if present
	if req.Config != nil && len(req.Config.Tools) > 0 {
		tools, err := convertTools(req.Config.Tools)
		if err != nil {
			return openai.ChatCompletionRequest{}, err
		}
		openaiReq.Tools = tools
	}

	// Apply config settings
	if req.Config != nil {
		if req.Config.Temperature != nil {
			openaiReq.Temperature = *req.Config.Temperature
		}
		if req.Config.MaxOutputTokens > 0 {
			openaiReq.MaxTokens = int(req.Config.MaxOutputTokens)
		}
		if req.Config.TopP != nil {
			openaiReq.TopP = *req.Config.TopP
		}
		if len(req.Config.StopSequences) > 0 {
			openaiReq.Stop = req.Config.StopSequences
		}

		// Handle system instruction
		if req.Config.SystemInstruction != nil {
			systemMsg := openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleSystem,
				Content: extractTextFromContent(req.Config.SystemInstruction),
			}
			openaiMessages = append([]openai.ChatCompletionMessage{systemMsg}, openaiMessages...)
			openaiReq.Messages = openaiMessages
		}

		// Handle JSON mode
		if req.Config.ResponseMIMEType == "application/json" {
			openaiReq.ResponseFormat = &openai.ChatCompletionResponseFormat{
				Type: openai.ChatCompletionResponseFormatTypeJSONObject,
			}
		}
	}

	return openaiReq, nil
}

func toOpenAIChatCompletionMessage(content *genai.Content) (openai.ChatCompletionMessage, error) {
	openaiMsg := openai.ChatCompletionMessage{
		Role: convertRoleToOpenAI(content.Role),
	}

	// Simple case: single text part
	if len(content.Parts) == 1 && content.Parts[0].Text != "" {
		openaiMsg.Content = content.Parts[0].Text
		return openaiMsg, nil
	}

	// Complex case: multiple parts or special part types
	var textContent string
	var toolCalls []openai.ToolCall
	var multiContent []openai.ChatMessagePart

	for _, part := range content.Parts {
		if part.Text != "" {
			if len(content.Parts) == 1 {
				textContent = part.Text
			} else {
				multiContent = append(multiContent, openai.ChatMessagePart{
					Type: openai.ChatMessagePartTypeText,
					Text: part.Text,
				})
			}
		}

		if part.FunctionCall != nil {
			argsJSON, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return openai.ChatCompletionMessage{}, fmt.Errorf("failed to marshal function args: %w", err)
			}
			toolCall := openai.ToolCall{
				ID:   part.FunctionCall.ID,
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: string(argsJSON),
				},
			}
			toolCalls = append(toolCalls, toolCall)
		}

		if part.FunctionResponse != nil {
			// Function responses become tool messages
			responseJSON, err := json.Marshal(part.FunctionResponse.Response)
			if err != nil {
				return openai.ChatCompletionMessage{}, fmt.Errorf("failed to marshal function response: %w", err)
			}
			openaiMsg.Role = openai.ChatMessageRoleTool
			openaiMsg.Content = string(responseJSON)
			openaiMsg.ToolCallID = part.FunctionResponse.ID
		}

		if part.InlineData != nil {
			base64Data := base64.StdEncoding.EncodeToString(part.InlineData.Data)
			imageURL := openai.ChatMessageImageURL{
				URL:    fmt.Sprintf("data:%s;base64,%s", part.InlineData.MIMEType, base64Data),
				Detail: openai.ImageURLDetailAuto,
			}
			multiContent = append(multiContent, openai.ChatMessagePart{
				Type:     openai.ChatMessagePartTypeImageURL,
				ImageURL: &imageURL,
			})
		}

		if part.FileData != nil {

			// OpenAI doesn't support file references directly, would need to download
			// For now, we'll skip or add as text description
		}
	}

	// Set content based on what we found
	if len(multiContent) > 0 {
		openaiMsg.MultiContent = multiContent
	} else if textContent != "" {
		openaiMsg.Content = textContent
	}

	if len(toolCalls) > 0 {
		openaiMsg.ToolCalls = toolCalls
	}

	return openaiMsg, nil
}

func convertChatCompletionResponse(resp *openai.ChatCompletionResponse) (*model.LLMResponse, error) {
	if len(resp.Choices) == 0 {
		return nil, ErrNoChoicesInResponse
	}

	choice := resp.Choices[0]
	content := &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{},
	}

	// Convert message content
	if choice.Message.Content != "" {
		content.Parts = append(content.Parts, &genai.Part{Text: choice.Message.Content})
	}

	// Convert tool calls
	for _, toolCall := range choice.Message.ToolCalls {
		if toolCall.Type == openai.ToolTypeFunction {
			content.Parts = append(content.Parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   toolCall.ID,
					Name: toolCall.Function.Name,
					Args: parseJSONArgs(toolCall.Function.Arguments),
				},
			})
		}
	}

	// Convert usage metadata
	var usageMetadata *genai.GenerateContentResponseUsageMetadata
	if resp.Usage.TotalTokens > 0 {
		usageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(resp.Usage.PromptTokens),
			CandidatesTokenCount: int32(resp.Usage.CompletionTokens),
			TotalTokenCount:      int32(resp.Usage.TotalTokens),
		}
		if resp.Usage.PromptTokensDetails != nil {
			usageMetadata.CachedContentTokenCount = int32(resp.Usage.PromptTokensDetails.CachedTokens)
		}
	}

	return &model.LLMResponse{
		Content:       content,
		UsageMetadata: usageMetadata,
		FinishReason:  convertFinishReason(string(choice.FinishReason)),
		TurnComplete:  true,
	}, nil
}

func convertTools(genaiTools []*genai.Tool) ([]openai.Tool, error) {
	var openaiTools []openai.Tool

	for _, genaiTool := range genaiTools {
		if genaiTool == nil {
			continue
		}

		// Convert function declarations
		for _, funcDecl := range genaiTool.FunctionDeclarations {
			if funcDecl == nil {
				// Only supports custom tools, not built-in tools
				panic("funcDecl is nil")
			}

			openaiTool := openai.Tool{
				Type: openai.ToolTypeFunction,
				Function: &openai.FunctionDefinition{
					Name:        funcDecl.Name,
					Description: funcDecl.Description,
					Parameters:  funcDecl.ParametersJsonSchema,
				},
			}
			openaiTools = append(openaiTools, openaiTool)
		}
	}

	return openaiTools, nil
}

func convertSchema(schema *genai.Schema) (map[string]any, error) {
	if schema == nil {
		return map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}, nil
	}

	result := make(map[string]any)

	// Convert type
	if schema.Type != genai.TypeUnspecified {
		result["type"] = convertSchemaType(schema.Type)
	}

	// Add description
	if schema.Description != "" {
		result["description"] = schema.Description
	}

	// Convert properties recursively
	if len(schema.Properties) > 0 {
		properties := make(map[string]any)
		for propName, propSchema := range schema.Properties {
			convertedProp, err := convertSchema(propSchema)
			if err != nil {
				return nil, err
			}
			properties[propName] = convertedProp
		}
		result["properties"] = properties
	}

	// Add required fields
	if len(schema.Required) > 0 {
		result["required"] = schema.Required
	}

	// Convert array items
	if schema.Items != nil {
		items, err := convertSchema(schema.Items)
		if err != nil {
			return nil, err
		}
		result["items"] = items
	}

	// Add enum if present
	if len(schema.Enum) > 0 {
		result["enum"] = schema.Enum
	}

	return result, nil
}

func convertSchemaType(t genai.Type) string {
	switch t {
	case genai.TypeString:
		return "string"
	case genai.TypeNumber:
		return "number"
	case genai.TypeInteger:
		return "integer"
	case genai.TypeBoolean:
		return "boolean"
	case genai.TypeArray:
		return "array"
	case genai.TypeObject:
		return "object"
	default:
		return "string"
	}
}

func convertRoleToOpenAI(role string) string {
	switch role {
	case "user":
		return openai.ChatMessageRoleUser
	case "model":
		return openai.ChatMessageRoleAssistant
	case "system":
		return openai.ChatMessageRoleSystem
	default:
		return openai.ChatMessageRoleUser
	}
}

func convertFinishReason(reason string) genai.FinishReason {
	switch reason {
	case "stop":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "tool_calls", "function_call":
		return genai.FinishReasonStop
	case "content_filter":
		return genai.FinishReasonSafety
	default:
		return genai.FinishReasonUnspecified
	}
}

func extractTextFromContent(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var texts []string
	for _, part := range content.Parts {
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return joinTexts(texts)
}

func joinTexts(texts []string) string {
	if len(texts) == 0 {
		return ""
	}
	if len(texts) == 1 {
		return texts[0]
	}
	result := ""
	for i, text := range texts {
		if i > 0 {
			result += "\n"
		}
		result += text
	}
	return result
}

func parseJSONArgs(argsJSON string) map[string]any {
	if argsJSON == "" {
		return make(map[string]any)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		// If parsing fails, return empty map
		return make(map[string]any)
	}
	return args
}
