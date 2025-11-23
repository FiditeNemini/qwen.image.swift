import Foundation

struct PhiloxRandom {
  private static let mul0: UInt32 = 0xD251_1F53
  private static let mul1: UInt32 = 0xCD9E_8D57
  private static let weyl0: UInt32 = 0x9E37_79B9
  private static let weyl1: UInt32 = 0xBB67_AE85
  private static let rounds: Int = 10

  private var counter: (UInt32, UInt32, UInt32, UInt32)
  private var key: (UInt32, UInt32)

  init(seed: UInt64, subsequence: UInt64 = 0, offset: UInt64 = 0) {
    self.key = (
      UInt32(truncatingIfNeeded: seed),
      UInt32(truncatingIfNeeded: seed >> 32)
    )
    let ctrAdvance = offset
    let ctr0 = UInt32(truncatingIfNeeded: ctrAdvance)
    let ctr1 = UInt32(truncatingIfNeeded: ctrAdvance >> 32)
    let ctr2 = UInt32(truncatingIfNeeded: subsequence)
    let ctr3 = UInt32(truncatingIfNeeded: subsequence >> 32)
    self.counter = (ctr0, ctr1, ctr2, ctr3)
    skip(offset: offset)
  }

  mutating func next() -> (UInt32, UInt32, UInt32, UInt32) {
    var ctr = counter
    var roundKey = key
    for _ in 0..<PhiloxRandom.rounds {
      ctr = PhiloxRandom.round(counter: ctr, key: roundKey)
      roundKey.0 &+= PhiloxRandom.weyl0
      roundKey.1 &+= PhiloxRandom.weyl1
    }
    incrementCounter()
    return ctr
  }

  private mutating func skip(offset: UInt64) {
    guard offset > 0 else { return }
    var remaining = offset
    while remaining > 0 {
      incrementCounter()
      remaining &-= 1
    }
  }

  private mutating func incrementCounter() {
    counter.0 &+= 1
    if counter.0 != 0 { return }
    counter.1 &+= 1
    if counter.1 != 0 { return }
    counter.2 &+= 1
    if counter.2 != 0 { return }
    counter.3 &+= 1
  }

  private static func round(
    counter: (UInt32, UInt32, UInt32, UInt32),
    key: (UInt32, UInt32)
  ) -> (UInt32, UInt32, UInt32, UInt32) {
    let (c0, c1, c2, c3) = counter
    let hi0 = mulhi(mul0, c0)
    let lo0 = mul0 &* c0
    let hi1 = mulhi(mul1, c2)
    let lo1 = mul1 &* c2
    let new0 = hi1 ^ c1 ^ key.0
    let new1 = lo1
    let new2 = hi0 ^ c3 ^ key.1
    let new3 = lo0
    return (new0, new1, new2, new3)
  }

  private static func mulhi(_ a: UInt32, _ b: UInt32) -> UInt32 {
    UInt32((UInt64(a) * UInt64(b)) >> 32)
  }
}

struct PhiloxNormalGenerator {
  private var engine: PhiloxRandom
  private var normalCache: Float32?
  private static let uniformEpsilon: Float32 = 1e-12

  init(seed: UInt64, subsequence: UInt64 = 0, offset: UInt64 = 0) {
    self.engine = PhiloxRandom(seed: seed, subsequence: subsequence, offset: offset / 4)
    normalCache = nil
  }

  mutating func generate(count: Int) -> [Float32] {
    var output: [Float32] = []
    output.reserveCapacity(count)
    while output.count < count {
      if let cached = normalCache {
        output.append(cached)
        normalCache = nil
        continue
      }
      let tuple = engine.next()
      let uniforms = [
        PhiloxNormalGenerator.toUniform(tuple.0),
        PhiloxNormalGenerator.toUniform(tuple.1),
        PhiloxNormalGenerator.toUniform(tuple.2),
        PhiloxNormalGenerator.toUniform(tuple.3)
      ]
      var idx = 0
      while idx + 1 < uniforms.count && output.count < count {
        let u1 = max(uniforms[idx], PhiloxNormalGenerator.uniformEpsilon)
        let u2 = uniforms[idx + 1]
        let radius = sqrt(-2.0 * log(u1))
        let theta = 2.0 * Float32.pi * u2
        let z0 = radius * cos(theta)
        let z1 = radius * sin(theta)
        output.append(z0)
        if output.count < count {
          output.append(z1)
        } else {
          normalCache = z1
        }
        idx += 2
      }
    }
    return output
  }

  private static func toUniform(_ value: UInt32) -> Float32 {
    // (value + 0.5) * 2^-32
    let inv: Float32 = 1.0 / 4_294_967_296.0
    return (Float32(value) + 0.5) * inv
  }
}
