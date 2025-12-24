import Foundation

struct LoraReference: Equatable {
  let repoId: String
  let revision: String
  let filePath: String?
}

enum LoraReferenceParser {
  static func parse(_ value: String) -> LoraReference? {
    if let urlReference = parseHuggingFaceURL(value) {
      return urlReference
    }
    if let repoReference = parseRepoSpecifier(value) {
      return repoReference
    }
    return nil
  }

  static func parseHuggingFaceURL(_ value: String) -> LoraReference? {
    guard let url = URL(string: value),
          let host = url.host?.lowercased(),
          host == "huggingface.co" || host == "www.huggingface.co" || host == "hf.co" else {
      return nil
    }
    let parts = url.path.split(separator: "/").map { String($0) }
    guard let markerIndex = parts.firstIndex(where: { $0 == "blob" || $0 == "resolve" || $0 == "raw" }) else {
      return nil
    }
    guard markerIndex >= 2 else {
      return nil
    }
    let repoId = parts[0..<markerIndex].joined(separator: "/")
    let revisionIndex = markerIndex + 1
    guard revisionIndex < parts.count else {
      return nil
    }
    let revision = parts[revisionIndex]
    let fileParts = parts.dropFirst(revisionIndex + 1)
    guard !fileParts.isEmpty else {
      return nil
    }
    let filePath = fileParts.joined(separator: "/")
    return LoraReference(repoId: repoId, revision: revision, filePath: filePath)
  }

  static func parseRepoSpecifier(_ value: String) -> LoraReference? {
    let parts = value.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
    if parts.count == 2 {
      let repoPart = String(parts[0])
      let filePart = String(parts[1])
      guard !repoPart.isEmpty, !filePart.isEmpty else {
        return nil
      }
      let (repoId, revision) = splitRepoRevision(repoPart)
      return LoraReference(repoId: repoId, revision: revision, filePath: filePart)
    }
    if value.contains("@") {
      let (repoId, revision) = splitRepoRevision(value)
      if repoId != value {
        return LoraReference(repoId: repoId, revision: revision, filePath: nil)
      }
    }
    return nil
  }

  static func splitRepoRevision(_ value: String) -> (String, String) {
    guard let atIndex = value.lastIndex(of: "@") else {
      return (value, "main")
    }
    let repoId = String(value[..<atIndex])
    let revision = String(value[value.index(after: atIndex)...])
    guard !repoId.isEmpty, !revision.isEmpty else {
      return (value, "main")
    }
    return (repoId, revision)
  }

  static func patterns(for filePath: String?) -> [String] {
    guard let filePath else {
      return ["*.safetensors", "**/*.safetensors"]
    }
    let trimmed = filePath.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    guard !trimmed.isEmpty else {
      return ["*.safetensors", "**/*.safetensors"]
    }
    if trimmed.contains("/") {
      return [trimmed]
    }
    return [trimmed, "**/\(trimmed)"]
  }
}
