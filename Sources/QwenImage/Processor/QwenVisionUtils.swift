import Foundation

enum QwenVisionError: Error {
  case extremeAspectRatio
}

enum QwenVisionUtils {

  static func smartResize(
    height: Int,
    width: Int,
    factor: Int,
    minPixels: Int,
    maxPixels: Int
  ) throws -> (height: Int, width: Int) {
    precondition(height > 0 && width > 0)
    precondition(factor > 0)
    precondition(minPixels > 0 && maxPixels >= minPixels)

    let longSide = max(height, width)
    let shortSide = min(height, width)
    if Double(longSide) / Double(shortSide) > 200.0 {
      throw QwenVisionError.extremeAspectRatio
    }

    func roundToMultiple(_ value: Double, _ factor: Int) -> Int {
      Int((value / Double(factor)).rounded(.toNearestOrEven)) * factor
    }

    func floorToMultiple(_ value: Double, _ factor: Int) -> Int {
      max(factor, Int(floor(value / Double(factor))) * factor)
    }

    func ceilToMultiple(_ value: Double, _ factor: Int) -> Int {
      Int(ceil(value / Double(factor))) * factor
    }

    let h0 = roundToMultiple(Double(height), factor)
    let w0 = roundToMultiple(Double(width), factor)

    var resizedHeight = h0
    var resizedWidth = w0

    let initialPixels = h0 * w0
    if initialPixels > maxPixels {
      let beta = sqrt(Double(height * width) / Double(maxPixels))
      resizedHeight = floorToMultiple(Double(height) / beta, factor)
      resizedWidth = floorToMultiple(Double(width) / beta, factor)
    } else if initialPixels < minPixels {
      let beta = sqrt(Double(minPixels) / Double(height * width))
      resizedHeight = ceilToMultiple(Double(height) * beta, factor)
      resizedWidth = ceilToMultiple(Double(width) * beta, factor)
    }

    return (resizedHeight, resizedWidth)
  }
}
