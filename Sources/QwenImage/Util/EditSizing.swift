import Foundation

public enum EditSizing {

  private static func roundToNearestMultiple(_ value: Double, multiple: Int) -> Int {
    guard multiple > 0 else {
      return max(1, Int(value.rounded(.toNearestOrEven)))
    }
    let scaled = value / Double(multiple)
    let rounded = scaled.rounded(.toNearestOrEven)
    return max(multiple, Int(rounded) * multiple)
  }

  public static func computeDimensions(
    referenceWidth: Int,
    referenceHeight: Int,
    targetArea: Int,
    multiple: Int = 16
  ) -> (width: Int, height: Int) {
    let rw = max(1, referenceWidth)
    let rh = max(1, referenceHeight)
    let area = max(1, targetArea)
    let aspect = Double(rw) / Double(rh)

    let widthEstimate = (Double(area) * aspect).squareRoot()
    let heightEstimate = widthEstimate / max(aspect, 1e-6)

    let width = roundToNearestMultiple(widthEstimate, multiple: multiple)
    let height = roundToNearestMultiple(heightEstimate, multiple: multiple)

    return (width, height)
  }

  public static func computeVAEDimensions(
    referenceWidth: Int,
    referenceHeight: Int,
    targetArea: Int = 1_048_576,
    multiple: Int = 32
  ) -> (width: Int, height: Int) {
    let rw = max(1, referenceWidth)
    let rh = max(1, referenceHeight)
    let aspect = Double(rw) / Double(rh)
    let area = max(1, targetArea)

    let widthEstimate = (Double(area) * aspect).squareRoot()
    let heightEstimate = widthEstimate / max(aspect, 1e-6)

    let width = roundToNearestMultiple(widthEstimate, multiple: multiple)
    let height = roundToNearestMultiple(heightEstimate, multiple: multiple)

    return (width, height)
  }

  public static func computeVisionConditionDimensions(
    referenceWidth: Int,
    referenceHeight: Int,
    targetArea: Int = 147_456,
    multiple: Int = 32
  ) -> (width: Int, height: Int) {
    let rw = max(1, referenceWidth)
    let rh = max(1, referenceHeight)
    let ratio = Double(rw) / Double(rh)
    let area = max(1, targetArea)

    let widthEstimate = (Double(area) * ratio).squareRoot()
    let heightEstimate = widthEstimate / max(ratio, 1e-6)

    let width = max(multiple, Int((widthEstimate / Double(multiple)).rounded(.toNearestOrEven)) * multiple)
    let height = max(multiple, Int((heightEstimate / Double(multiple)).rounded(.toNearestOrEven)) * multiple)

    return (width, height)
  }
}
