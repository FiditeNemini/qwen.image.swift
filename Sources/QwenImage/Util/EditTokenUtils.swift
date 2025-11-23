import Foundation
import MLX

enum EditTokenUtilsError: Error {
  case missingPlaceholder(row: Int)
  case placeholderCountMismatch(row: Int, expected: Int, found: Int)
  case invalidPlaceholderSpan(row: Int)
  case invalidRepeatCounts
  case missingImageToken
}

enum EditTokenUtils {
  struct PlaceholderInfo {
    let offsets: [[Int]]
    let lengths: [[Int]]
  }

  static func expandVisionPlaceholders(
    batch: QwenTokenBatch,
    padTokenId: Int,
    imageTokenId: Int,
    visionStartTokenId: Int?,
    visionEndTokenId: Int?,
    repeatCounts: [Int]
  ) throws -> (batch: QwenTokenBatch, startOffsets: [[Int]], spanLengths: [[Int]]) {
    guard repeatCounts.allSatisfy({ $0 >= 0 }) else {
      throw EditTokenUtilsError.invalidRepeatCounts
    }
    let rows = batch.inputIds.dim(0)
    let cols = batch.inputIds.dim(1)
    if repeatCounts.isEmpty {
      let empty: [[Int]] = Array(repeating: [], count: rows)
      return (batch, empty, empty)
    }
    let ids = batch.inputIds.asType(.int32)
    let mask = batch.attentionMask.asType(.int32)
    MLX.eval(ids, mask)
    let idsArray = ids.asArray(Int32.self)
    let maskArray = mask.asArray(Int32.self)

    var newIdRows: [[Int32]] = Array(repeating: [], count: rows)
    var newMaskRows: [[Int32]] = Array(repeating: [], count: rows)
    var startOffsets: [[Int]] = Array(repeating: [], count: rows)
    var maxLen = 0

    for r in 0..<rows {
      let base = r * cols
      var active: [Int32] = []
      var activeMask: [Int32] = []
      for c in 0..<cols {
        let mv = maskArray[base + c]
        if mv == 0 { break }
        active.append(idsArray[base + c])
        activeMask.append(mv)
      }

      guard let startToken = visionStartTokenId, let endToken = visionEndTokenId else {
        newIdRows[r] = active
        newMaskRows[r] = activeMask
        if active.count > maxLen { maxLen = active.count }
        continue
      }

      let startId = Int32(startToken)
      let endId = Int32(endToken)
      var seq: [Int32] = []
      var seqMask: [Int32] = []
      var placeholderIndex = 0
      var idx = 0

      while idx < active.count {
        let token = active[idx]
        let tokenMask = activeMask[idx]
        seq.append(token)
        seqMask.append(tokenMask)

        if token == startId {
          guard placeholderIndex < repeatCounts.count else {
            QwenLogger.editTokens.warning("placeholderCountMismatch row=\(r) index=\(placeholderIndex) repeatCounts=\(repeatCounts)")
            throw EditTokenUtilsError.placeholderCountMismatch(
              row: r,
              expected: repeatCounts.count,
              found: placeholderIndex + 1
            )
          }

          let insertStart = seq.count
          startOffsets[r].append(insertStart)

          let repeatCount = repeatCounts[placeholderIndex]
          if repeatCount > 0 {
            seq.append(contentsOf: Array(repeating: Int32(imageTokenId), count: repeatCount))
            seqMask.append(contentsOf: Array(repeating: Int32(1), count: repeatCount))
          }

          var search = idx + 1
          while search < active.count, active[search] != endId {
            search += 1
          }
          guard search < active.count else {
            QwenLogger.editTokens.warning("invalidPlaceholderSpan row=\(r) startIndex=\(idx) placeholderIndex=\(placeholderIndex)")
            throw EditTokenUtilsError.invalidPlaceholderSpan(row: r)
          }

          seq.append(active[search])
          seqMask.append(activeMask[search])
          placeholderIndex += 1
          idx = search
        }

        idx += 1
      }

      if placeholderIndex != repeatCounts.count {
        QwenLogger.editTokens.warning("missingPlaceholder row=\(r) found=\(placeholderIndex) expected=\(repeatCounts.count)")
        throw EditTokenUtilsError.missingPlaceholder(row: r)
      }

      newIdRows[r] = seq
      newMaskRows[r] = seqMask
      if seq.count > maxLen { maxLen = seq.count }
    }

    for i in 0..<rows {
      let diff = maxLen - newIdRows[i].count
      if diff > 0 {
        newIdRows[i].append(contentsOf: Array(repeating: Int32(padTokenId), count: diff))
        newMaskRows[i].append(contentsOf: Array(repeating: 0, count: diff))
      }
    }

    let idsFlat = newIdRows.flatMap { $0 }
    let maskFlat = newMaskRows.flatMap { $0 }
    let idTensor = MLXArray(idsFlat.map { Float32($0) }, [rows, maxLen]).asType(.int32)
    let maskTensor = MLXArray(maskFlat.map { Float32($0) }, [rows, maxLen]).asType(.int32)
    QwenLogger.editTokens.debug("startOffsets=\(startOffsets)")
    let spanLengths = Array(repeating: repeatCounts, count: rows)
    return (QwenTokenBatch(inputIds: idTensor, attentionMask: maskTensor), startOffsets, spanLengths)
  }

  static func locateVisionPlaceholderTokens(
    batch: QwenTokenBatch,
    imageTokenId: Int
  ) throws -> PlaceholderInfo {
    let rows = batch.inputIds.dim(0)
    let cols = batch.inputIds.dim(1)
    let ids = batch.inputIds.asType(.int32)
    let mask = batch.attentionMask.asType(.int32)
    MLX.eval(ids, mask)
    let idsArray = ids.asArray(Int32.self)
    let maskArray = mask.asArray(Int32.self)
    var offsets: [[Int]] = Array(repeating: [], count: rows)
    var lengths: [[Int]] = Array(repeating: [], count: rows)
    let token = Int32(imageTokenId)

    for r in 0..<rows {
      let base = r * cols
      var index = 0
      while index < cols {
        if maskArray[base + index] == 0 {
          break
        }
        if idsArray[base + index] == token {
          offsets[r].append(index)
          lengths[r].append(1)
        }
        index += 1
      }
    }

    return PlaceholderInfo(offsets: offsets, lengths: lengths)
  }
}
