// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "qwen.image.swift",
  platforms: [.macOS(.v14), .iOS(.v16)],
  products: [
    .library(name: "QwenImage", targets: ["QwenImage"]),
    .executable(name: "QwenImageCLI", targets: ["QwenImageCLI"])
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.29.1")),
    .package(
      url: "https://github.com/huggingface/swift-transformers",
      .upToNextMinor(from: "0.1.21")
    ),
    .package(url: "https://github.com/apple/swift-log.git", from: "1.6.4")
  ],
  targets: [
    .target(
      name: "QwenImage",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXFast", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXOptimizers", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "Transformers", package: "swift-transformers"),
        .product(name: "Logging", package: "swift-log")
      ],
      path: "Sources/QwenImage"
    ),
    .executableTarget(
      name: "QwenImageCLI",
      dependencies: ["QwenImage"],
      path: "Sources/QwenImageCLI"
    ),
    .testTarget(
      name: "QwenImageTests",
      dependencies: ["QwenImage"],
      path: "Tests/QwenImageTests"
    ),
    .testTarget(
      name: "QwenImageCLITests",
      dependencies: ["QwenImageCLI"],
      path: "Tests/QwenImageCLITests"
    )
  ]
)
