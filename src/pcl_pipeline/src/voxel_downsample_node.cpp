// 間引き: 一辺 leaf_size の立方体ごとに、中の点を重心 1 点にまとめる
#include <memory>
#include <vector>

#include "pcl_pipeline/filters.hpp"
#include "pcl_pipeline/node_util.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#ifdef PCL_PIPELINE_WITH_CUDA
#include "pcl_pipeline/voxel_cuda.hpp"
#endif

namespace pcl_pipeline {

class VoxelDownsample : public rclcpp::Node {
public:
  explicit VoxelDownsample(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("voxel_downsample", options), stats_(this) {
    p_.leaf = static_cast<float>(declare_parameter("leaf_size", 0.2));
    // M5: use_cuda:=true なら CUDA 版を使う(CUDA なしでビルドしたときは CPU 版のまま)
    if (declare_parameter("use_cuda", false)) {
#ifdef PCL_PIPELINE_WITH_CUDA
      if (VoxelCuda::available()) {
        cuda_ = std::make_unique<VoxelCuda>();
      } else {
        RCLCPP_WARN(get_logger(), "use_cuda=true but no CUDA device; using CPU");
      }
#else
      RCLCPP_WARN(get_logger(), "use_cuda=true but built without CUDA; using CPU");
#endif
    }
    RCLCPP_INFO(get_logger(), "voxel on %s", use_gpu() ? "GPU (CUDA)" : "CPU");
    pub_ = create_publisher<PointCloud2>("downsampled/points", cloud_qos(this));
    sub_ = create_subscription<PointCloud2>(
      "cropped/points", cloud_qos(this), [this](PointCloud2::ConstSharedPtr msg) { on_cloud(*msg); });
  }

private:
  void on_cloud(const PointCloud2 & msg) {
    Stopwatch sw;
    from_msg(msg, in_);  // in_・out_・scratch_ は使い回す(M7)
    run();
    auto out_msg = to_msg(out_, msg.header);  // header(センサーの時刻)はそのまま引き継ぐ
    stats_.add(sw.ms(), msg.header.stamp, in_.size(), out_.size());
    pub_->publish(std::move(out_msg));
  }

  bool use_gpu() const {
#ifdef PCL_PIPELINE_WITH_CUDA
    return cuda_ != nullptr;
#else
    return false;
#endif
  }
  // in_ を間引いて、out_ に入れる
  void run() {
#ifdef PCL_PIPELINE_WITH_CUDA
    if (cuda_) {
      out_ = cuda_->run(in_, p_.leaf);
      return;
    }
#endif
    voxel_downsample(in_, p_.leaf, scratch_, out_);
  }

  struct Params {
    float leaf;
  } p_;
  std::vector<Point> in_, out_;  // 毎フレーム確保し直さない(同じノードのコールバックは同時に動かない)
  VoxelScratch scratch_;
#ifdef PCL_PIPELINE_WITH_CUDA
  std::unique_ptr<VoxelCuda> cuda_;
#endif
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr sub_;
  StageStats stats_;
};

}  // namespace pcl_pipeline

// 部品(component)として登録する。1 プロセスにまとめるとき(M3)も、単独の実行ファイルとしても使える
RCLCPP_COMPONENTS_REGISTER_NODE(pcl_pipeline::VoxelDownsample)
