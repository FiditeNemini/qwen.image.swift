import Foundation
import QwenImage
import MLX
import Logging

// MARK: - Session Configuration

/// Configuration for ImagePipelineSession resource management policies.
public struct ImagePipelineSessionConfiguration: Sendable {
  /// Whether to release encoders after encoding prompts (before generation).
  /// Default is true to save memory during the denoising loop.
  public var releaseEncodersAfterEncoding: Bool

  /// Maximum number of cached prompt embeddings.
  public var maxCachedEmbeddings: Int

  /// GPU cache limit in bytes. If nil, no limit is set by the session.
  public var gpuCacheLimit: Int?

  /// Default configuration with memory-saving defaults.
  public static let `default` = ImagePipelineSessionConfiguration(
    releaseEncodersAfterEncoding: true,
    maxCachedEmbeddings: 10,
    gpuCacheLimit: nil
  )

  /// Configuration optimized for fast prompt iteration (keeps encoders loaded).
  public static let fastPromptIteration = ImagePipelineSessionConfiguration(
    releaseEncodersAfterEncoding: false,
    maxCachedEmbeddings: 20,
    gpuCacheLimit: nil
  )

  /// Configuration optimized for memory-constrained environments.
  public static let lowMemory = ImagePipelineSessionConfiguration(
    releaseEncodersAfterEncoding: true,
    maxCachedEmbeddings: 3,
    gpuCacheLimit: 2 * 1024 * 1024 * 1024  // 2GB
  )

  public init(
    releaseEncodersAfterEncoding: Bool = true,
    maxCachedEmbeddings: Int = 10,
    gpuCacheLimit: Int? = nil
  ) {
    self.releaseEncodersAfterEncoding = releaseEncodersAfterEncoding
    self.maxCachedEmbeddings = maxCachedEmbeddings
    self.gpuCacheLimit = gpuCacheLimit
  }
}

// MARK: - Session

/// Session wrapper for QwenImagePipeline with policy management.
///
/// This actor wraps a QwenImagePipeline and provides:
/// - Automatic prompt embedding caching with proper cache keys
/// - Configurable encoder release policy
/// - Thread-safe access to the pipeline
///
/// Usage:
/// ```swift
/// let pipeline = QwenImagePipeline(config: .textToImage)
/// pipeline.setBaseDirectory(modelPath)
/// try pipeline.prepareTokenizer(from: modelPath)
/// try pipeline.prepareTextEncoder(from: modelPath)
///
/// let session = ImagePipelineSession(
///   pipeline: pipeline,
///   modelId: "Qwen/Qwen-Image",
///   revision: "main"
/// )
///
/// let params = GenerationParameters(prompt: "A cat", width: 1024, height: 1024, steps: 30)
/// let pixels = try await session.generate(parameters: params, model: modelConfig)
/// ```
public actor ImagePipelineSession {
  private let pipeline: QwenImagePipeline
  private let embeddingsCache: PromptEmbeddingsCache
  private let modelId: String
  private let revision: String
  private let configuration: ImagePipelineSessionConfiguration
  private var logger = Logger(label: "qwen.image.session")

  /// Create a new session wrapping a pipeline.
  /// - Parameters:
  ///   - pipeline: The pipeline to wrap. Should have tokenizer and text encoder loaded.
  ///   - modelId: Model identifier for cache keys.
  ///   - revision: Model revision for cache keys.
  ///   - configuration: Session configuration.
  public init(
    pipeline: QwenImagePipeline,
    modelId: String,
    revision: String = "main",
    configuration: ImagePipelineSessionConfiguration = .default
  ) {
    self.pipeline = pipeline
    self.modelId = modelId
    self.revision = revision
    self.configuration = configuration
    self.embeddingsCache = PromptEmbeddingsCache(maxEntries: configuration.maxCachedEmbeddings)

    // Apply GPU cache limit if specified
    if let limit = configuration.gpuCacheLimit {
      GPU.set(cacheLimit: limit)
    }
  }

  // MARK: - Generation

  /// Generate an image with automatic caching and resource management.
  ///
  /// This method:
  /// 1. Checks the cache for existing prompt embeddings
  /// 2. Encodes the prompt if not cached (requires tokenizer/textEncoder)
  /// 3. Caches the encoding for future use
  /// 4. Optionally releases encoders based on configuration
  /// 5. Runs generation with the cached/computed encoding
  ///
  /// - Parameters:
  ///   - parameters: Generation parameters.
  ///   - model: Model configuration.
  ///   - maxPromptLength: Maximum prompt length (defaults to model's max).
  ///   - seed: Random seed for reproducibility.
  /// - Returns: Generated pixel array.
  public func generate(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil
  ) async throws -> MLXArray {
    let maxLength = maxPromptLength ?? model.maxSequenceLength

    // Get or compute guidance encoding
    let encoding = try await guidanceEncoding(
      prompt: parameters.prompt,
      negativePrompt: parameters.negativePrompt,
      maxLength: maxLength
    )

    // Release encoders if configured to do so
    if configuration.releaseEncodersAfterEncoding {
      pipeline.releaseEncoders()
      logger.debug("Released encoders after encoding")
    }

    // Generate using the policy-free overload
    return try pipeline.generatePixels(
      parameters: parameters,
      model: model,
      guidanceEncoding: encoding,
      seed: seed
    )
  }

  // MARK: - Encoding

  /// Get or compute guidance encoding with caching.
  ///
  /// - Parameters:
  ///   - prompt: The prompt text.
  ///   - negativePrompt: Optional negative prompt.
  ///   - maxLength: Maximum sequence length.
  /// - Returns: The guidance encoding (cached or freshly computed).
  public func guidanceEncoding(
    prompt: String,
    negativePrompt: String?,
    maxLength: Int
  ) async throws -> QwenGuidanceEncoding {
    let cacheKey = PromptEmbeddingsCacheKey(
      modelId: modelId,
      revision: revision,
      maxLength: maxLength,
      prompt: prompt,
      negativePrompt: negativePrompt
    )

    // Check cache first
    if let cached = await embeddingsCache.get(key: cacheKey) {
      logger.debug("Cache hit for prompt encoding")
      return cached
    }

    logger.debug("Cache miss, encoding prompt")

    try pipeline.reloadTokenizer()
    try pipeline.reloadTextEncoder()

    let encoding = try pipeline.encodeGuidancePrompts(
      prompt: prompt,
      negativePrompt: negativePrompt,
      maxLength: maxLength
    )

    await embeddingsCache.set(key: cacheKey, value: encoding)
    return encoding
  }

  // MARK: - Lifecycle Management

  /// Explicitly release encoder components to free memory.
  public func releaseEncoders() {
    pipeline.releaseEncoders()
    logger.debug("Encoders released explicitly")
  }

  /// Release only the text encoder.
  public func releaseTextEncoder() {
    pipeline.releaseTextEncoder()
  }

  /// Release only the vision tower.
  public func releaseVisionTower() {
    pipeline.releaseVisionTower()
  }

  /// Reload the text encoder (if weights directory is set).
  public func reloadTextEncoder() throws {
    try pipeline.reloadTextEncoder()
  }

  /// Reload the tokenizer (if weights directory is set).
  public func reloadTokenizer() throws {
    try pipeline.reloadTokenizer()
  }

  // MARK: - Cache Management

  /// Clear all cached prompt embeddings.
  public func clearCache() async {
    await embeddingsCache.invalidateAll()
    logger.debug("Embeddings cache cleared")
  }

  /// Invalidate cache entries for the current model.
  public func invalidateModelCache() async {
    await embeddingsCache.invalidateForModel(modelId, revision: revision)
  }

  /// The current number of cached embeddings.
  public var cacheCount: Int {
    get async {
      await embeddingsCache.count
    }
  }

  // MARK: - Status

  /// Check if the text encoder is currently loaded.
  public var isTextEncoderLoaded: Bool {
    pipeline.isTextEncoderLoaded
  }

  /// Check if the tokenizer is currently loaded.
  public var isTokenizerLoaded: Bool {
    pipeline.isTokenizerLoaded
  }

  /// Check if the vision tower is currently loaded.
  public var isVisionTowerLoaded: Bool {
    pipeline.isVisionTowerLoaded
  }

  /// Check if the UNet is currently loaded.
  public var isUNetLoaded: Bool {
    pipeline.isUNetLoaded
  }

  /// Check if the VAE is currently loaded.
  public var isVAELoaded: Bool {
    pipeline.isVAELoaded
  }
}

// MARK: - Sync Wrapper

extension ImagePipelineSession {
  /// Synchronous wrapper for cache lookup (non-isolated).
  /// Use this when you need to check the cache without async context.
  nonisolated public func hasCachedEncoding(
    prompt: String,
    negativePrompt: String?,
    maxLength: Int
  ) async -> Bool {
    let cacheKey = PromptEmbeddingsCacheKey(
      modelId: modelId,
      revision: revision,
      maxLength: maxLength,
      prompt: prompt,
      negativePrompt: negativePrompt
    )
    return await embeddingsCache.contains(key: cacheKey)
  }
}
