import Logging

enum QwenLogger {
  static let pipeline = Logger(label: "qwen.image.pipeline")
  static let vision = Logger(label: "qwen.image.vision")
  static let tokenizer = Logger(label: "qwen.image.tokenizer")
  static let editTokens = Logger(label: "qwen.image.edit-tokens")
  static let text = Logger(label: "qwen.image.text")
}
