import XCTest
@testable import QwenImageCLI

final class LoraReferenceParserTests: XCTestCase {
  func testParsesHuggingFaceFileURL() {
    let value = "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/blob/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    let reference = LoraReferenceParser.parse(value)
    XCTAssertEqual(reference?.repoId, "lightx2v/Qwen-Image-Edit-2511-Lightning")
    XCTAssertEqual(reference?.revision, "main")
    XCTAssertEqual(reference?.filePath, "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
  }

  func testParsesRepoSpecifierWithFile() {
    let value = "lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    let reference = LoraReferenceParser.parse(value)
    XCTAssertEqual(reference?.repoId, "lightx2v/Qwen-Image-Edit-2511-Lightning")
    XCTAssertEqual(reference?.revision, "main")
    XCTAssertEqual(reference?.filePath, "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
  }

  func testParsesRepoSpecifierWithRevisionAndFile() {
    let value = "lightx2v/Qwen-Image-Edit-2511-Lightning@v1.0:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    let reference = LoraReferenceParser.parse(value)
    XCTAssertEqual(reference?.repoId, "lightx2v/Qwen-Image-Edit-2511-Lightning")
    XCTAssertEqual(reference?.revision, "v1.0")
    XCTAssertEqual(reference?.filePath, "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
  }

  func testPlainRepoIdReturnsNil() {
    let reference = LoraReferenceParser.parse("lightx2v/Qwen-Image-Edit-2511-Lightning")
    XCTAssertNil(reference)
  }

  func testPatternsForFileSelection() {
    XCTAssertEqual(LoraReferenceParser.patterns(for: nil), ["*.safetensors", "**/*.safetensors"])
    XCTAssertEqual(
      LoraReferenceParser.patterns(for: "adapter.safetensors"),
      ["adapter.safetensors", "**/adapter.safetensors"]
    )
    XCTAssertEqual(
      LoraReferenceParser.patterns(for: "subdir/adapter.safetensors"),
      ["subdir/adapter.safetensors"]
    )
  }
}
