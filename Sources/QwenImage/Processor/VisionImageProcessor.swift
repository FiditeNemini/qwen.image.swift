import Foundation
import MLX

#if canImport(CoreGraphics)
import CoreGraphics
#endif

enum VisionImageProcessor {}

#if canImport(CoreGraphics)
extension VisionImageProcessor {
  private struct BicubicWeights {
    let start: Int
    let coefficients: [Float32]
  }

  private struct KernelContribution {
    var index: Int
    var weight: Double
  }

  static func resizeAndNormalize(
    image: CGImage,
    width: Int,
    height: Int,
    mean: [Float],
    std: [Float],
    rescaleFactor: Float = 1.0,
    addBatchDimension: Bool = true,
    dtype: DType = .float32,
    intermediateWidth: Int? = nil,
    intermediateHeight: Int? = nil
  ) throws -> MLXArray {
    let srcWidth = image.width
    let srcHeight = image.height
    let bytesPerPixel = 4

    let stageWidth = intermediateWidth ?? srcWidth
    let stageHeight = intermediateHeight ?? srcHeight

    let planar: [Float32]
    let sourceWidth: Int
    let sourceHeight: Int
    if stageWidth != srcWidth || stageHeight != srcHeight {
      let stageArray = try QwenImageIO.resizedPixelArray(
        from: image,
        width: stageWidth,
        height: stageHeight,
        addBatchDimension: false,
        dtype: .float32
      )
      MLX.eval(stageArray)
      let floats = stageArray.asArray(Float32.self)
      planar = floats
      sourceWidth = stageWidth
      sourceHeight = stageHeight
    } else {
      var raw = [UInt8](repeating: 0, count: srcWidth * srcHeight * bytesPerPixel)
      let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
      let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
      let drawn = raw.withUnsafeMutableBytes { ptr -> Bool in
        guard let baseAddress = ptr.baseAddress else { return false }
        guard let context = CGContext(
          data: baseAddress,
          width: srcWidth,
          height: srcHeight,
          bitsPerComponent: 8,
          bytesPerRow: srcWidth * bytesPerPixel,
          space: colorSpace,
          bitmapInfo: bitmapInfo
        ) else {
          return false
        }
        let rect = CGRect(x: 0, y: 0, width: srcWidth, height: srcHeight)
        context.draw(image, in: rect)
        return true
      }
      guard drawn else {
        throw QwenVisionPreprocessorError.imageResizeFailed
      }
      planar = planarRGB(from: raw, width: srcWidth, height: srcHeight)
      sourceWidth = srcWidth
      sourceHeight = srcHeight
    }

    let resized = torchvisionBicubicResize(
      planar: planar,
      srcWidth: sourceWidth,
      srcHeight: sourceHeight,
      dstWidth: width,
      dstHeight: height
    )

    let channelSize = width * height
    var tensorValues = [Float32](repeating: 0, count: channelSize * 3)
    for channel in 0..<3 {
      let meanValue = mean[min(channel, mean.count - 1)]
      let stdValue = std[min(channel, std.count - 1)]
      let offset = channel * channelSize
      for idx in 0..<channelSize {
        let value = resized[offset + idx]
        tensorValues[offset + idx] = (value * rescaleFactor - meanValue) / stdValue
      }
    }

    var shape = [3, height, width]
    if addBatchDimension {
      shape.insert(1, at: 0)
    }
    var tensor = MLXArray(tensorValues, shape)
    if dtype != .float32 {
      tensor = tensor.asType(dtype)
    }
    return tensor
  }

  private static func planarRGB(
    from rgbaBytes: [UInt8],
    width: Int,
    height: Int
  ) -> [Float32] {
    let pixelCount = width * height
    var planar = [Float32](repeating: 0, count: pixelCount * 3)
    for index in 0..<pixelCount {
      let offset = index * 4
      planar[index] = Float32(rgbaBytes[offset]) / 255.0
      planar[index + pixelCount] = Float32(rgbaBytes[offset + 1]) / 255.0
      planar[index + 2 * pixelCount] = Float32(rgbaBytes[offset + 2]) / 255.0
    }
    return planar
  }

  private static func torchvisionBicubicResize(
    planar: [Float32],
    srcWidth: Int,
    srcHeight: Int,
    dstWidth: Int,
    dstHeight: Int
  ) -> [Float32] {
    guard srcWidth > 0, srcHeight > 0, dstWidth > 0, dstHeight > 0 else {
      return [Float32](repeating: 0, count: max(1, 3 * max(dstWidth, 1) * max(dstHeight, 1)))
    }
    if srcWidth == dstWidth && srcHeight == dstHeight {
      return planar
    }
    let channels = 3
    let srcSize = srcWidth * srcHeight
    precondition(planar.count == channels * srcSize, "Expected planar RGB data with 3 channels.")

    let weightsX = computeTorchvisionAAContributions(srcLength: srcWidth, dstLength: dstWidth)
    let weightsY = computeTorchvisionAAContributions(srcLength: srcHeight, dstLength: dstHeight)

    let tempRowCount = srcHeight
    var temp = [Float32](repeating: 0, count: channels * tempRowCount * dstWidth)

    for channel in 0..<channels {
      let srcChannelOffset = channel * srcSize
      let tempChannelOffset = channel * tempRowCount * dstWidth
      for row in 0..<srcHeight {
        let srcRowOffset = srcChannelOffset + row * srcWidth
        let tempRowOffset = tempChannelOffset + row * dstWidth
        for x in 0..<dstWidth {
          let contrib = weightsX[x]
          var sum: Float32 = 0
          if contrib.coefficients.isEmpty {
            let sampleIndex = min(max(contrib.start, 0), srcWidth - 1)
            sum = planar[srcRowOffset + sampleIndex]
          } else {
            for (tapIndex, weight) in contrib.coefficients.enumerated() {
              let sourceIndex = contrib.start + tapIndex
              if sourceIndex < 0 || sourceIndex >= srcWidth {
                continue
              }
              sum += planar[srcRowOffset + sourceIndex] * weight
            }
          }
          temp[tempRowOffset + x] = sum
        }
      }
    }

    let dstSize = dstWidth * dstHeight
    var output = [Float32](repeating: 0, count: channels * dstSize)
    for channel in 0..<channels {
      let tempChannelOffset = channel * tempRowCount * dstWidth
      let dstChannelOffset = channel * dstSize
      for y in 0..<dstHeight {
        let contrib = weightsY[y]
        let dstRowOffset = dstChannelOffset + y * dstWidth
        for x in 0..<dstWidth {
          var sum: Float32 = 0
          if contrib.coefficients.isEmpty {
            let sampleIndex = min(max(contrib.start, 0), srcHeight - 1)
            sum = temp[tempChannelOffset + sampleIndex * dstWidth + x]
          } else {
            for (tapIndex, weight) in contrib.coefficients.enumerated() {
              let sourceIndex = contrib.start + tapIndex
              if sourceIndex < 0 || sourceIndex >= srcHeight {
                continue
              }
              sum += temp[tempChannelOffset + sourceIndex * dstWidth + x] * weight
            }
          }
          output[dstRowOffset + x] = sum
        }
      }
    }

    return output
  }

  private static func computeTorchvisionAAContributions(
    srcLength: Int,
    dstLength: Int
  ) -> [BicubicWeights] {
    computeTorchvisionCubicContributions(
      srcLength: srcLength,
      dstLength: dstLength,
      alignCorners: false,
      antialias: true
    )
  }

  private static func computeTorchvisionCubicContributions(
    srcLength: Int,
    dstLength: Int,
    alignCorners: Bool,
    antialias: Bool
  ) -> [BicubicWeights] {
    guard srcLength > 0, dstLength > 0 else { return [] }
    let scale = areaPixelComputeScale(
      inputSize: srcLength,
      outputSize: dstLength,
      alignCorners: alignCorners,
      scaleOverride: nil
    )
    return computeIndexRangesWeights(
      inputSize: srcLength,
      outputSize: dstLength,
      scale: scale,
      interpSize: 4,
      antialias: antialias,
      alignCorners: alignCorners
    )
  }

  private static func computeIndexRangesWeights(
    inputSize: Int,
    outputSize: Int,
    scale: Float32,
    interpSize: Int,
    antialias: Bool,
    alignCorners: Bool
  ) -> [BicubicWeights] {
    let halfKernel = Float32(interpSize) * Float32(0.5)
    let support: Float32
    let maxInterpSize: Int
    if antialias {
      if scale >= Float32(1.0) {
        support = halfKernel * scale
      } else {
        support = halfKernel
      }
      let ceilSupport = Int(support.rounded(.up))
      maxInterpSize = max(1, ceilSupport * 2 + 1)
    } else {
      support = halfKernel
      maxInterpSize = interpSize
    }

    var contributions: [BicubicWeights] = []
    contributions.reserveCapacity(outputSize)
    for dstIndex in 0..<outputSize {
      var weights = [Float32](repeating: 0, count: maxInterpSize)
      var xmin = 0
      var xsize = 0
      if antialias {
        computeIndicesMinSizeWeightsAA(
          dstIndex: dstIndex,
          inputSize: inputSize,
          scale: scale,
          support: support,
          maxInterpSize: maxInterpSize,
          filter: { torchvisionAAFilter($0, antialias: true) },
          xmin: &xmin,
          xsize: &xsize,
          weights: &weights
        )
      } else {
        computeIndicesMinSizeWeightsNoAA(
          dstIndex: dstIndex,
          inputSize: inputSize,
          scale: scale,
          maxInterpSize: maxInterpSize,
          alignCorners: alignCorners,
          filter: { torchvisionAAFilter($0, antialias: false) },
          xmin: &xmin,
          xsize: &xsize,
          weights: &weights
        )
      }

      if xsize <= 0 {
        let safeIndex = clampIndex(
          areaPixelComputeSourceIndex(
            scale: scale,
            dstIndex: dstIndex,
            alignCorners: alignCorners,
            cubic: true
          ),
          upperBound: inputSize
        )
        contributions.append(BicubicWeights(start: safeIndex, coefficients: [1]))
        continue
      }

      let taps: [Float32]
      if xsize == weights.count {
        taps = weights
      } else {
        taps = Array(weights.prefix(xsize))
      }
      contributions.append(BicubicWeights(start: xmin, coefficients: taps))
    }
    return contributions
  }

  private static func computeIndicesMinSizeWeightsAA(
    dstIndex: Int,
    inputSize: Int,
    scale: Float32,
    support: Float32,
    maxInterpSize: Int,
    filter: (Float32) -> Float32,
    xmin: inout Int,
    xsize: inout Int,
    weights: inout [Float32]
  ) {
    let center = scale * (Float32(dstIndex) + Float32(0.5))
    let invScale = scale >= Float32(1.0) ? Float32(1.0) / scale : Float32(1.0)
    var minIndex = truncToInt(center - support + Float32(0.5))
    var maxIndex = truncToInt(center + support + Float32(0.5))
    if minIndex < 0 { minIndex = 0 }
    if maxIndex > inputSize { maxIndex = inputSize }
    var size = max(0, maxIndex - minIndex)
    if size > maxInterpSize {
      size = maxInterpSize
    }

    xmin = minIndex
    xsize = size
    guard size > 0 else {
      if inputSize > 0 {
        xmin = min(max(minIndex, 0), inputSize - 1)
        xsize = 1
        weights[0] = 1
      }
      for tap in xsize..<weights.count {
        weights[tap] = 0
      }
      return
    }

    var total: Float32 = 0
    for tap in 0..<size {
      let distance = (Float32(tap + minIndex) - center + Float32(0.5)) * invScale
      let weight = filter(distance)
      weights[tap] = weight
      total += weight
    }
    if total != 0 {
      let invTotal = Float32(1.0) / total
      for tap in 0..<size {
        weights[tap] *= invTotal
      }
    }
    if size < weights.count {
      for tap in size..<weights.count {
        weights[tap] = 0
      }
    }
  }

  private static func computeIndicesMinSizeWeightsNoAA(
    dstIndex: Int,
    inputSize: Int,
    scale: Float32,
    maxInterpSize: Int,
    alignCorners: Bool,
    filter: (Float32) -> Float32,
    xmin: inout Int,
    xsize: inout Int,
    weights: inout [Float32]
  ) {
    let support = Int(Float32(maxInterpSize) * Float32(0.5))
    let realIndex = areaPixelComputeSourceIndex(
      scale: scale,
      dstIndex: dstIndex,
      alignCorners: alignCorners,
      cubic: maxInterpSize > 2
    )
    let (inputIndex, lambda) = guardIndexAndLambda(realInputIndex: realIndex, inputSize: inputSize)
    let unboundMin = inputIndex - support + 1
    let unboundMax = inputIndex + support + 1

    var minIndex = max(unboundMin, 0)
    var maxIndex = min(unboundMax, inputSize)
    var size = maxIndex - minIndex
    if size > maxInterpSize {
      size = maxInterpSize
    }
    xmin = minIndex
    xsize = size

    if size <= 0 {
      if inputSize > 0 {
        xmin = min(max(inputIndex, 0), inputSize - 1)
        xsize = 1
        weights[0] = 1
      }
      for tap in xsize..<weights.count {
        weights[tap] = 0
      }
      return
    }

    var weightIndex = 0
    for tap in 0..<maxInterpSize {
      weights[tap] = 0
      let value = filter(Float32(tap + 1 - support) - lambda)
      if unboundMin + tap <= 0 {
        weightIndex = 0
      } else if unboundMin + tap >= inputSize - 1 {
        weightIndex = max(size - 1, 0)
      }
      weights[min(weightIndex, size - 1)] += value
      weightIndex += 1
    }
  }

  private static func torchvisionAAFilter(_ distance: Float32, antialias: Bool) -> Float32 {
    let a: Float32 = antialias ? -0.5 : -0.75
    let x = abs(distance)
    if x < Float32(1.0) {
      return cubicConvolution1(x: x, a: a)
    }
    if x < Float32(2.0) {
      return cubicConvolution2(x: x, a: a)
    }
    return 0
  }

  private static func cubicConvolution1(x: Float32, a: Float32) -> Float32 {
    return ((a + Float32(2.0)) * x - (a + Float32(3.0))) * x * x + Float32(1.0)
  }

  private static func cubicConvolution2(x: Float32, a: Float32) -> Float32 {
    return ((a * x - Float32(5.0) * a) * x + Float32(8.0) * a) * x - Float32(4.0) * a
  }

  private static func computeScaleValue(
    scaleOverride: Float32?,
    inputSize: Int,
    outputSize: Int
  ) -> Float32 {
    if let provided = scaleOverride, provided > 0 {
      return Float32(1.0) / provided
    }
    guard outputSize > 0 else { return 0 }
    return Float32(inputSize) / Float32(outputSize)
  }

  private static func areaPixelComputeScale(
    inputSize: Int,
    outputSize: Int,
    alignCorners: Bool,
    scaleOverride: Float32?
  ) -> Float32 {
    if alignCorners {
      if outputSize > 1 {
        return Float32(inputSize - 1) / Float32(outputSize - 1)
      } else {
        return 0
      }
    } else {
      return computeScaleValue(
        scaleOverride: scaleOverride,
        inputSize: inputSize,
        outputSize: outputSize
      )
    }
  }

  private static func areaPixelComputeSourceIndex(
    scale: Float32,
    dstIndex: Int,
    alignCorners: Bool,
    cubic: Bool
  ) -> Float32 {
    if alignCorners {
      return scale * Float32(dstIndex)
    } else {
      let src = scale * (Float32(dstIndex) + Float32(0.5)) - Float32(0.5)
      if !cubic && src < 0 {
        return 0
      }
      return src
    }
  }

  private static func guardIndexAndLambda(
    realInputIndex: Float32,
    inputSize: Int
  ) -> (index: Int, lambda: Float32) {
    if inputSize <= 0 {
      return (0, 0)
    }
    let floored = Int(floorf(realInputIndex))
    let clamped = min(floored, inputSize - 1)
    let frac = min(
      max(realInputIndex - Float32(clamped), Float32(0)),
      Float32(1.0)
    )
    return (clamped, frac)
  }

  private static func truncToInt(_ value: Float32) -> Int {
    return Int(value.rounded(.towardZero))
  }

  private static func clampIndex(_ value: Float32, upperBound: Int) -> Int {
    guard upperBound > 0 else { return 0 }
    let floored = Int(floorf(value))
    if floored < 0 { return 0 }
    if floored >= upperBound { return upperBound - 1 }
    return floored
  }

}
#else
extension VisionImageProcessor {
  static func resizeAndNormalize(
    image: CGImage,
    width: Int,
    height: Int,
    mean: [Float],
    std: [Float],
    rescaleFactor: Float = 1.0,
    addBatchDimension: Bool = true,
    dtype: DType = .float32,
    intermediateWidth: Int? = nil,
    intermediateHeight: Int? = nil
  ) throws -> MLXArray {
    throw QwenVisionPreprocessorError.unsupportedPlatform
  }
}
#endif
