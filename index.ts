/**
 * Corporate Gateway Provider Extension
 *
 * Connects pi to Claude models hosted via the internal API gateway.
 *
 * Azure Foundry uses the Anthropic Messages API but requires:
 *   - A different base URL (the internal gateway endpoint)
 *   - api-key <key> header (instead of x-api-key or Authorization: Bearer)
 *
 * Environment variables:
 *   CORP_GATEWAY_API_KEY   - Your gateway API key (required)
 *   CORP_GATEWAY_BASE_URL  - Your gateway endpoint base URL (required)
 *                             e.g. https://grove-gateway-prod.azure-api.net/grove-foundry-prod/anthropic
 *
 * Usage:
 *   CORP_GATEWAY_API_KEY=your-key pi
 *   # Then /model and select corp-gateway/claude-opus-5
 *                              or corp-gateway/claude-sonnet-5
 *                              or corp-gateway/claude-opus-4-6
 */

import Anthropic from "@anthropic-ai/sdk";
import type { ContentBlockParam, MessageCreateParamsStreaming } from "@anthropic-ai/sdk/resources/messages.js";
import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Context,
	calculateCost,
	createAssistantMessageEventStream,
	type ImageContent,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type StopReason,
	type TextContent,
	type ThinkingContent,
	type Tool,
	type ToolResultMessage,
} from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

// =============================================================================
// Message Conversion (adapted from the custom-provider-anthropic example)
// =============================================================================

function sanitizeSurrogates(text: string): string {
	return text.replace(/[\uD800-\uDFFF]/g, "\uFFFD");
}

function convertContentBlocks(
	content: (TextContent | ImageContent)[],
): string | Array<{ type: "text"; text: string } | { type: "image"; source: any }> {
	const hasImages = content.some((c) => c.type === "image");
	if (!hasImages) {
		return sanitizeSurrogates(content.map((c) => (c as TextContent).text).join("\n"));
	}

	const blocks = content.map((block) => {
		if (block.type === "text") {
			return { type: "text" as const, text: sanitizeSurrogates(block.text) };
		}
		return {
			type: "image" as const,
			source: {
				type: "base64" as const,
				media_type: block.mimeType,
				data: block.data,
			},
		};
	});

	if (!blocks.some((b) => b.type === "text")) {
		blocks.unshift({ type: "text" as const, text: "(see attached image)" });
	}

	return blocks;
}

function convertMessages(messages: Message[]): any[] {
	const params: any[] = [];

	for (let i = 0; i < messages.length; i++) {
		const msg = messages[i];

		if (msg.role === "user") {
			if (typeof msg.content === "string") {
				if (msg.content.trim()) {
					params.push({ role: "user", content: sanitizeSurrogates(msg.content) });
				}
			} else {
				const blocks: ContentBlockParam[] = msg.content.map((item) =>
					item.type === "text"
						? { type: "text" as const, text: sanitizeSurrogates(item.text) }
						: {
								type: "image" as const,
								source: { type: "base64" as const, media_type: item.mimeType as any, data: item.data },
							},
				);
				if (blocks.length > 0) {
					params.push({ role: "user", content: blocks });
				}
			}
		} else if (msg.role === "assistant") {
			const blocks: ContentBlockParam[] = [];
			for (const block of msg.content) {
				if (block.type === "text" && block.text.trim()) {
					blocks.push({ type: "text", text: sanitizeSurrogates(block.text) });
				} else if (block.type === "thinking" && block.thinking.trim()) {
					if ((block as ThinkingContent).thinkingSignature) {
						blocks.push({
							type: "thinking" as any,
							thinking: sanitizeSurrogates(block.thinking),
							signature: (block as ThinkingContent).thinkingSignature!,
						});
					} else {
						blocks.push({ type: "text", text: sanitizeSurrogates(block.thinking) });
					}
				} else if (block.type === "toolCall") {
					blocks.push({
						type: "tool_use",
						id: block.id,
						name: block.name,
						input: block.arguments,
					});
				}
			}
			if (blocks.length > 0) {
				params.push({ role: "assistant", content: blocks });
			}
		} else if (msg.role === "toolResult") {
			const toolResults: any[] = [];
			toolResults.push({
				type: "tool_result",
				tool_use_id: msg.toolCallId,
				content: convertContentBlocks(msg.content),
				is_error: msg.isError,
			});

			let j = i + 1;
			while (j < messages.length && messages[j].role === "toolResult") {
				const nextMsg = messages[j] as ToolResultMessage;
				toolResults.push({
					type: "tool_result",
					tool_use_id: nextMsg.toolCallId,
					content: convertContentBlocks(nextMsg.content),
					is_error: nextMsg.isError,
				});
				j++;
			}
			i = j - 1;
			params.push({ role: "user", content: toolResults });
		}
	}

	// Add cache control to last user message
	if (params.length > 0) {
		const last = params[params.length - 1];
		if (last.role === "user" && Array.isArray(last.content)) {
			const lastBlock = last.content[last.content.length - 1];
			if (lastBlock) {
				lastBlock.cache_control = { type: "ephemeral" };
			}
		}
	}

	return params;
}

function convertTools(tools: Tool[]): any[] {
	return tools.map((tool) => ({
		name: tool.name,
		description: tool.description,
		input_schema: {
			type: "object",
			properties: (tool.parameters as any).properties || {},
			required: (tool.parameters as any).required || [],
		},
	}));
}

function mapStopReason(reason: string): StopReason {
	switch (reason) {
		case "end_turn":
		case "pause_turn":
		case "stop_sequence":
			return "stop";
		case "max_tokens":
			return "length";
		case "tool_use":
			return "toolUse";
		default:
			return "error";
	}
}

// =============================================================================
// Streaming Implementation
// =============================================================================

function streamCorpGateway(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();

	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};

		try {
			const apiKey = options?.apiKey ?? "";

			// Azure Foundry requires Authorization: Bearer, which the Anthropic SDK
			// sends when using authToken (vs apiKey which sends x-api-key).
			const client = new Anthropic({
				baseURL: model.baseUrl,
				apiKey: null as any,
				dangerouslyAllowBrowser: true,
				defaultHeaders: {
					accept: "application/json",					
					"anthropic-version": "2023-06-01",
					"anthropic-beta": "fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14",
					"anthropic-dangerous-direct-browser-access": "true",
					"api-key": apiKey,
					"x-api-key": null,
				},
			});

			// Build request params
			const params: MessageCreateParamsStreaming = {
				model: model.id,
				messages: convertMessages(context.messages),
				max_tokens: options?.maxTokens || Math.floor(model.maxTokens / 3),
				stream: true,
			};

			// System prompt
			if (context.systemPrompt) {
				params.system = [
					{
						type: "text",
						text: sanitizeSurrogates(context.systemPrompt),
						cache_control: { type: "ephemeral" },
					},
				];
			}

			if (context.tools) {
				params.tools = convertTools(context.tools);
			}

			// Handle thinking/reasoning.
			// Gen-5+ models (claude-*-5, claude-opus-5, etc.) use the new
			// "adaptive" thinking API with output_config.effort, while older
			// models use the "enabled" API with budget_tokens.
			if (options?.reasoning && model.reasoning) {
				const usesAdaptiveThinking = /claude-[a-z]+-5/.test(model.id) || model.id.startsWith("claude-opus-5");
				if (usesAdaptiveThinking) {
					// Map pi reasoning levels → Anthropic effort values
					const effortMap: Record<string, string> = {
						minimal: "low",
						low: "low",
						medium: "medium",
						high: "high",
						xhigh: "high",
					};
					const effort = effortMap[options.reasoning] ?? "medium";
					(params as any).thinking = { type: "adaptive" };
					(params as any).output_config = { effort };
				} else {
					const defaultBudgets: Record<string, number> = {
						minimal: 1024,
						low: 4096,
						medium: 10240,
						high: 20480,
					};
					const customBudget = options.thinkingBudgets?.[options.reasoning as keyof typeof options.thinkingBudgets];
					params.thinking = {
						type: "enabled",
						budget_tokens: customBudget ?? defaultBudgets[options.reasoning] ?? 10240,
					};
				}
			}

			const anthropicStream = client.messages.stream({ ...params }, { signal: options?.signal });
			stream.push({ type: "start", partial: output });

			type Block = (ThinkingContent | TextContent | (import("@mariozechner/pi-ai").ToolCall & { partialJson: string })) & {
				index: number;
			};
			const blocks = output.content as Block[];

			for await (const event of anthropicStream) {
				if (event.type === "message_start") {
					output.usage.input = event.message.usage.input_tokens || 0;
					output.usage.output = event.message.usage.output_tokens || 0;
					output.usage.cacheRead = (event.message.usage as any).cache_read_input_tokens || 0;
					output.usage.cacheWrite = (event.message.usage as any).cache_creation_input_tokens || 0;
					output.usage.totalTokens =
						output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
					calculateCost(model, output.usage);
				} else if (event.type === "content_block_start") {
					if (event.content_block.type === "text") {
						output.content.push({ type: "text", text: "", index: event.index } as any);
						stream.push({ type: "text_start", contentIndex: output.content.length - 1, partial: output });
					} else if (event.content_block.type === "thinking") {
						output.content.push({
							type: "thinking",
							thinking: "",
							thinkingSignature: "",
							index: event.index,
						} as any);
						stream.push({ type: "thinking_start", contentIndex: output.content.length - 1, partial: output });
					} else if (event.content_block.type === "tool_use") {
						output.content.push({
							type: "toolCall",
							id: event.content_block.id,
							name: event.content_block.name,
							arguments: {},
							partialJson: "",
							index: event.index,
						} as any);
						stream.push({ type: "toolcall_start", contentIndex: output.content.length - 1, partial: output });
					}
				} else if (event.type === "content_block_delta") {
					const index = blocks.findIndex((b) => b.index === event.index);
					const block = blocks[index];
					if (!block) continue;

					if (event.delta.type === "text_delta" && block.type === "text") {
						block.text += event.delta.text;
						stream.push({ type: "text_delta", contentIndex: index, delta: event.delta.text, partial: output });
					} else if (event.delta.type === "thinking_delta" && block.type === "thinking") {
						block.thinking += event.delta.thinking;
						stream.push({
							type: "thinking_delta",
							contentIndex: index,
							delta: event.delta.thinking,
							partial: output,
						});
					} else if (event.delta.type === "input_json_delta" && block.type === "toolCall") {
						(block as any).partialJson += event.delta.partial_json;
						try {
							block.arguments = JSON.parse((block as any).partialJson);
						} catch {}
						stream.push({
							type: "toolcall_delta",
							contentIndex: index,
							delta: event.delta.partial_json,
							partial: output,
						});
					} else if (event.delta.type === "signature_delta" && block.type === "thinking") {
						block.thinkingSignature = (block.thinkingSignature || "") + (event.delta as any).signature;
					}
				} else if (event.type === "content_block_stop") {
					const index = blocks.findIndex((b) => b.index === event.index);
					const block = blocks[index];
					if (!block) continue;

					delete (block as any).index;
					if (block.type === "text") {
						stream.push({ type: "text_end", contentIndex: index, content: block.text, partial: output });
					} else if (block.type === "thinking") {
						stream.push({ type: "thinking_end", contentIndex: index, content: block.thinking, partial: output });
					} else if (block.type === "toolCall") {
						try {
							block.arguments = JSON.parse((block as any).partialJson);
						} catch {}
						delete (block as any).partialJson;
						stream.push({ type: "toolcall_end", contentIndex: index, toolCall: block, partial: output });
					}
				} else if (event.type === "message_delta") {
					if ((event.delta as any).stop_reason) {
						output.stopReason = mapStopReason((event.delta as any).stop_reason);
					}
					output.usage.input = (event.usage as any).input_tokens || 0;
					output.usage.output = (event.usage as any).output_tokens || 0;
					output.usage.cacheRead = (event.usage as any).cache_read_input_tokens || 0;
					output.usage.cacheWrite = (event.usage as any).cache_creation_input_tokens || 0;
					output.usage.totalTokens =
						output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
					calculateCost(model, output.usage);
				}
			}

			if (options?.signal?.aborted) {
				throw new Error("Request was aborted");
			}

			stream.push({ type: "done", reason: output.stopReason as "stop" | "length" | "toolUse", message: output });
			stream.end();
		} catch (error) {
			for (const block of output.content) delete (block as any).index;
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end();
		}
	})();

	return stream;
}

// =============================================================================
// Model Discovery
// =============================================================================

/**
 * Fallback model list used at startup and if discovery fails.
 * Keep this in sync with what you know is deployed.
 */
const FALLBACK_MODELS: Model<Api>[] = [
	{
		id: "claude-opus-5",
		name: "Claude Opus 5 (Azure Foundry)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
		contextWindow: 1000000,
		maxTokens: 32000,
	},
	{
		id: "claude-fable-5",
		name: "Claude Fable 5 (Azure Foundry)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 10, output: 50, cacheRead: 1, cacheWrite: 12.5 },
		contextWindow: 1000000,
		maxTokens: 32000,
	},
	{
		id: "claude-sonnet-5",
		name: "Claude Sonnet 5 (Azure Foundry)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 2, output: 10, cacheRead: 0.2, cacheWrite: 4 },
		contextWindow: 1000000,
		maxTokens: 64000,
	},
	{
		id: "claude-opus-4-8",
		name: "Claude Opus 4.8 (Azure Foundry)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
		contextWindow: 1000000,
		maxTokens: 32000,
	},
	{
		id: "claude-opus-4-7",
		name: "Claude Opus 4.7 (Azure Foundry)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
		contextWindow: 1000000,
		maxTokens: 32000,
	},
	{
		id: "claude-opus-4-6",
		name: "Claude Opus 4.6 (Azure Foundry)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
		contextWindow: 1000000,
		maxTokens: 32000,
	},
	{
		id: "claude-sonnet-4-6",
		name: "Claude Sonnet 4.6 (Azure Foundry)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
		contextWindow: 1000000,
		maxTokens: 64000,
	},
] as unknown as Model<Api>[];

/**
 * Metadata we know about specific model ID prefixes, used to populate
 * cost/capability fields for models returned by the gateway's /v1/models.
 */
const KNOWN_MODEL_META: Array<{
	prefix: string;
	reasoning: boolean;
	cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
	contextWindow: number;
	maxTokens: number;
}> = [
	// ----- Generation 5 -----
	{
		prefix: "claude-opus-5",
		reasoning: true,
		cost: { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
		contextWindow: 1000000,
		maxTokens: 32000,
	},
	{
		prefix: "claude-fable-5",
		reasoning: true,
		cost: { input: 10, output: 50, cacheRead: 1, cacheWrite: 12.5 },
		contextWindow: 1000000,
		maxTokens: 32000,
	},
	{
		prefix: "claude-sonnet-5",
		reasoning: true,
		cost: { input: 2, output: 10, cacheRead: 0.2, cacheWrite: 4 },
		contextWindow: 1000000,
		maxTokens: 64000,
	},
	// ----- Opus 4 (specific versions before general prefix) -----
	{
		prefix: "claude-opus-4-8",
		reasoning: true,
		cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
		contextWindow: 1000000,
		maxTokens: 32000,
	},
	{
		prefix: "claude-opus-4-7",
		reasoning: true,
		cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
		contextWindow: 1000000,
		maxTokens: 32000,
	},
	{
		prefix: "claude-opus-4-6",
		reasoning: true,
		cost: { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
		contextWindow: 1000000,
		maxTokens: 32000,
	},
	// General Opus 4 catch-all (any future point releases)
	{
		prefix: "claude-opus-4",
		reasoning: true,
		cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
		contextWindow: 1000000,
		maxTokens: 32000,
	},
	// ----- Sonnet 4 -----
	{
		prefix: "claude-sonnet-4",
		reasoning: true,
		cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
		contextWindow: 1000000,
		maxTokens: 64000,
	},
	// ----- Haiku (unchanged) -----
	{
		prefix: "claude-haiku-3-5",
		reasoning: false,
		cost: { input: 0.8, output: 4, cacheRead: 0.08, cacheWrite: 1 },
		contextWindow: 200000,
		maxTokens: 8192,
	},
	{
		prefix: "claude-haiku-3",
		reasoning: false,
		cost: { input: 0.25, output: 1.25, cacheRead: 0.03, cacheWrite: 0.3 },
		contextWindow: 200000,
		maxTokens: 4096,
	},
];

function metaForModel(id: string) {
	return (
		KNOWN_MODEL_META.find((m) => id.startsWith(m.prefix)) ?? {
			reasoning: false,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 200000,
			maxTokens: 16384,
		}
	);
}

function buildModelDef(id: string, displayName?: string): Model<Api> {
	const meta = metaForModel(id);
	return {
		id,
		name: `${displayName ?? id} (Azure Foundry)`,
		input: ["text", "image"],
		...meta,
	} as unknown as Model<Api>;
}

async function discoverModels(baseUrl: string, apiKey: string): Promise<Model<Api>[]> {
	const client = new Anthropic({
		baseURL: baseUrl,
		apiKey: null as any,
		dangerouslyAllowBrowser: true,
		defaultHeaders: {
			accept: "application/json",
			"anthropic-version": "2023-06-01",
			"anthropic-dangerous-direct-browser-access": "true",
			"api-key": apiKey,
			"x-api-key": null,
		},
	});

	const page = await client.models.list();
	const models: Model<Api>[] = [];
	for (const m of page.data) {
		models.push(buildModelDef(m.id, (m as any).display_name));
	}
	return models;
}

// =============================================================================
// Extension Entry Point
// =============================================================================

export default function (pi: ExtensionAPI) {
	const baseUrl = process.env.CORP_GATEWAY_BASE_URL;
	if (!baseUrl) {
		console.error(
			"[corp-gateway] CORP_GATEWAY_BASE_URL is not set. " +
			"Set it to your gateway endpoint, e.g. https://grove-gateway-prod.azure-api.net/grove-foundry-prod/anthropic"
		);
		return;
	}

	// Register immediately with the fallback list so the provider is always
	// available from the first moment pi starts up.
	pi.registerProvider("corp-gateway", {
		baseUrl,
		apiKey: "CORP_GATEWAY_API_KEY",
		api: "corp-gateway-anthropic",
		models: FALLBACK_MODELS,
		streamSimple: streamCorpGateway,
	});

	// Helper that runs discovery and re-registers with the live model list.
	async function refreshModels(notify?: (msg: string, level: "info" | "error" | "success") => void) {
		const apiKey = process.env.CORP_GATEWAY_API_KEY;
		if (!apiKey) {
			notify?.("CORP_GATEWAY_API_KEY is not set — cannot discover models", "error");
			return;
		}
		try {
			const discovered = await discoverModels(baseUrl, apiKey);
			if (discovered.length === 0) {
				notify?.("[corp-gateway] /v1/models returned an empty list — keeping fallback models", "error");
				return;
			}
			pi.registerProvider("corp-gateway", {
				baseUrl,
				apiKey: "CORP_GATEWAY_API_KEY",
				api: "corp-gateway-anthropic",
				models: discovered,
				streamSimple: streamCorpGateway,
			});
			notify?.(
				`[corp-gateway] Discovered ${discovered.length} model(s): ${discovered.map((m) => m.id).join(", ")}`,
				"success",
			);
		} catch (err) {
			const msg = err instanceof Error ? err.message : String(err);
			notify?.(
				`[corp-gateway] Model discovery failed (keeping fallback list): ${msg}`,
				"error",
			);
		}
	}

	// Kick off discovery in the background on startup. If it succeeds the
	// provider is silently re-registered with the real model list; if it fails
	// (e.g. the gateway doesn't expose /v1/models) we stay on the fallback.
	refreshModels();

	// Manual refresh command for when you want to see what changed or force an
	// update mid-session: /corp-gateway-refresh-models
	pi.registerCommand("corp-gateway-refresh-models", {
		description: "Re-query the corp gateway for available models and update the provider list",
		handler: async (_args, ctx) => {
			ctx.ui.notify("Querying corp gateway for available models…", "info");
			await refreshModels((msg, level) => ctx.ui.notify(msg, level));
		},
	});
}
