import Foundation
import MLX
import MLXNN

final class QwenEmbedRope: Module {
  let theta: Int
  let axesDimensions: [Int]
  let scaleRope: Bool

  private let positiveCosValues: [MLXArray]
  private let positiveSinValues: [MLXArray]
  private let negativeCosValues: [MLXArray]
  private let negativeSinValues: [MLXArray]

  init(theta: Int, axesDimensions: [Int], scaleRope: Bool) {
    precondition(axesDimensions.count == 3, "Expected [frame, height, width] dimensions")
    self.theta = theta
    self.axesDimensions = axesDimensions
    self.scaleRope = scaleRope

    let maxIndex = 4096
    let positiveIndices = Array(0..<maxIndex)
    let negativeIndices = positiveIndices.reversed().map { -($0 + 1) }

    var posCos: [MLXArray] = []
    var posSin: [MLXArray] = []
    var negCos: [MLXArray] = []
    var negSin: [MLXArray] = []

    for dimension in axesDimensions {
      let (pc, ps) = QwenEmbedRope.ropeParameters(indices: positiveIndices, dimension: dimension, theta: theta)
      let (nc, ns) = QwenEmbedRope.ropeParameters(indices: negativeIndices, dimension: dimension, theta: theta)
      posCos.append(pc)
      posSin.append(ps)
      negCos.append(nc)
      negSin.append(ns)
    }

    self.positiveCosValues = posCos
    self.positiveSinValues = posSin
    self.negativeCosValues = negCos
    self.negativeSinValues = negSin
  }

  func callAsFunction(
    videoSegments: [(Int, Int, Int)],
    textSequenceLengths: [Int]
  ) -> (MLXArray, MLXArray) {
    precondition(!videoSegments.isEmpty, "At least one image segment is required")
    var cosSegments: [MLXArray] = []
    var sinSegments: [MLXArray] = []
    var maxVideoIndex = 0

    for (segmentIndex, (frame, height, width)) in videoSegments.enumerated() {
      let (segmentCos, segmentSin) = buildVideoFrequencies(
        frame: frame,
        height: height,
        width: width,
        frameIndexOffset: segmentIndex
      )
      cosSegments.append(segmentCos)
      sinSegments.append(segmentSin)

      let candidate: Int
      if scaleRope {
        candidate = max(height / 2, width / 2)
      } else {
        candidate = max(height, width)
      }
      if candidate > maxVideoIndex {
        maxVideoIndex = candidate
      }
    }

    let imgCos = cosSegments.count == 1 ? cosSegments[0] : MLX.concatenated(cosSegments, axis: 0)
    let imgSin = sinSegments.count == 1 ? sinSegments[0] : MLX.concatenated(sinSegments, axis: 0)

    let textLength = textSequenceLengths.max() ?? 0

    let cosFull = MLX.concatenated(positiveCosValues, axis: 1)
    let sinFull = MLX.concatenated(positiveSinValues, axis: 1)

    let start = maxVideoIndex
    let end = start + textLength
    let textCos = cosFull[start..<end, 0...]
    let textSin = sinFull[start..<end, 0...]

    return (
      rotationMatrix(from: imgCos, sin: imgSin),
      rotationMatrix(from: textCos, sin: textSin)
    )
  }

  func callAsFunction(
    videoFHW: (Int, Int, Int),
    textSequenceLengths: [Int]
  ) -> (MLXArray, MLXArray) {
    callAsFunction(videoSegments: [videoFHW], textSequenceLengths: textSequenceLengths)
  }

  private static func ropeParameters(
    indices: [Int],
    dimension: Int,
    theta: Int
  ) -> (MLXArray, MLXArray) {
    precondition(dimension % 2 == 0, "RoPE dimension must be even")
    let halfDim = dimension / 2
    var cosine: [Float32] = []
    var sine: [Float32] = []

    let scales = stride(from: 0, to: dimension, by: 2).map { Float($0) / Float(dimension) }
    let omega = scales.map { 1.0 / pow(Float(theta), $0) }

    for idx in indices {
      let value = Float(idx)
      for w in omega {
        let angle = value * w
        cosine.append(cos(angle))
        sine.append(sin(angle))
      }
    }

    let cosArray = MLXArray(cosine, [indices.count, halfDim])
    let sinArray = MLXArray(sine, [indices.count, halfDim])
    return (cosArray.asType(.float32), sinArray.asType(.float32))
  }

  private func buildVideoFrequencies(
    frame: Int,
    height: Int,
    width: Int,
    frameIndexOffset: Int
  ) -> (MLXArray, MLXArray) {
    let dimF = positiveCosValues[0].dim(1)
    let dimH = positiveCosValues[1].dim(1)
    let dimW = positiveCosValues[2].dim(1)

    let frameStart = frameIndexOffset
    let frameEnd = frameIndexOffset + frame
    let cosF = positiveCosValues[0][frameStart..<frameEnd, 0...].reshaped(frame, 1, 1, dimF)
    let sinF = positiveSinValues[0][frameStart..<frameEnd, 0...].reshaped(frame, 1, 1, dimF)

    let (cosHArray, sinHArray): (MLXArray, MLXArray)
    if scaleRope {
      let half = height / 2
      let positive = positiveCosValues[1][0..<half, 0...]
      let positiveSin = positiveSinValues[1][0..<half, 0...]
      let negativeLength = height - half
      let negStart = negativeCosValues[1].dim(0) - negativeLength
      let negative = negativeCosValues[1][negStart..<(negStart + negativeLength), 0...]
      let negativeSin = negativeSinValues[1][negStart..<(negStart + negativeLength), 0...]
      cosHArray = MLX.concatenated([negative, positive], axis: 0)
      sinHArray = MLX.concatenated([negativeSin, positiveSin], axis: 0)
    } else {
      cosHArray = positiveCosValues[1][0..<height, 0...]
      sinHArray = positiveSinValues[1][0..<height, 0...]
    }

    let cosH = cosHArray.reshaped(1, height, 1, dimH)
    let sinH = sinHArray.reshaped(1, height, 1, dimH)

    let (cosWArray, sinWArray): (MLXArray, MLXArray)
    if scaleRope {
      let half = width / 2
      let positive = positiveCosValues[2][0..<half, 0...]
      let positiveSin = positiveSinValues[2][0..<half, 0...]
      let negativeLength = width - half
      let negStart = negativeCosValues[2].dim(0) - negativeLength
      let negative = negativeCosValues[2][negStart..<(negStart + negativeLength), 0...]
      let negativeSin = negativeSinValues[2][negStart..<(negStart + negativeLength), 0...]
      cosWArray = MLX.concatenated([negative, positive], axis: 0)
      sinWArray = MLX.concatenated([negativeSin, positiveSin], axis: 0)
    } else {
      cosWArray = positiveCosValues[2][0..<width, 0...]
      sinWArray = positiveSinValues[2][0..<width, 0...]
    }

    let cosW = cosWArray.reshaped(1, 1, width, dimW)
    let sinW = sinWArray.reshaped(1, 1, width, dimW)

    let cos = MLX.concatenated(
      [
        tiled(cosF, repetitions: [1, height, width, 1]),
        tiled(cosH, repetitions: [frame, 1, width, 1]),
        tiled(cosW, repetitions: [frame, height, 1, 1])
      ],
      axis: -1
    )

    let sin = MLX.concatenated(
      [
        tiled(sinF, repetitions: [1, height, width, 1]),
        tiled(sinH, repetitions: [frame, 1, width, 1]),
        tiled(sinW, repetitions: [frame, height, 1, 1])
      ],
      axis: -1
    )

    let cosFlat = cos.reshaped(frame * height * width, cos.dim(3))
    let sinFlat = sin.reshaped(frame * height * width, sin.dim(3))
    return (cosFlat, sinFlat)
  }

  private func rotationMatrix(from cos: MLXArray, sin: MLXArray) -> MLXArray {
    let row0 = MLX.stacked([cos, -sin], axis: -1)
    let row1 = MLX.stacked([sin, cos], axis: -1)
    let rot = MLX.stacked([row0, row1], axis: -2)
    var expanded = rot.asType(.float32)
    expanded = expanded.expandedDimensions(axis: 0)
    expanded = expanded.expandedDimensions(axis: 0)
    return expanded
  }
}
