// M7: CPU 側で「毎フレーム確保して捨てる」のと「先に確保して使い回す」を、1 つの処理ごとに比べる(ROS は使わない)。
//   取り込み: メッセージ(バイト列)→ vector<Point>(from_msg 相当)
//     A 今までの形: vector<Point>(n) + memcpy(0 で埋めてから、すぐ上書きする)
//     B 0 で埋めない new Point[n] + memcpy
//     C 使い回し: resize + memcpy(今の from_msg)
//   間引き: voxel_downsample(約 33,000 点 → 約 11,000 点)
//     V0 今までの形: 毎回 unordered_map を作る(この実装は、M7 の前の filters.cpp と同じ)
//     V1 unordered_map を使い回す(clear + reserve)
//     V2 作業領域を毎回作る平らなハッシュ表(voxel_downsample(in, leaf))
//     V3 平らなハッシュ表を使い回す(今の voxel_downsample(in, leaf, scratch, out))
//   1 フレーム分: 取り込み + 間引き
// 点群は sensor_sim と同じ疑似 LiDAR を 50 個(ノイズが違うので、点の数も毎回少し違う)。
// 使い方: alloc_bench [iterations=3000]   各条件で、空打ち 300 回のあと測る。p50 / p99 / p99.9 / 最大を出す。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include "pcl_pipeline/filters.hpp"

using namespace pcl_pipeline;

namespace {

inline uint64_t cell_key(int32_t ix, int32_t iy, int32_t iz) {
  constexpr int32_t kOffset = 1 << 20;
  constexpr uint64_t kMask = (1u << 21) - 1;
  return ((static_cast<uint64_t>(ix + kOffset) & kMask) << 42) |
         ((static_cast<uint64_t>(iy + kOffset) & kMask) << 21) |
         (static_cast<uint64_t>(iz + kOffset) & kMask);
}
inline int32_t cell_of(float v, float size) { return static_cast<int32_t>(std::floor(v / size)); }
inline uint64_t key_of(const Point & p, float leaf) {
  return cell_key(cell_of(p.x, leaf), cell_of(p.y, leaf), cell_of(p.z, leaf));
}

struct Acc {
  double x = 0, y = 0, z = 0, i = 0;
  uint32_t n = 0;
};
using Map = std::unordered_map<uint64_t, Acc>;

// V0・V1: M7 の前の実装(unordered_map)。cells を呼び出し側が持てば、使い回しになる。
void voxel_with_map(const std::vector<Point> & in, float leaf, Map & cells,
                    std::vector<Point> & out) {
  cells.clear();
  cells.reserve(in.size());
  for (const auto & pt : in) {
    auto & a = cells[key_of(pt, leaf)];
    a.x += pt.x;
    a.y += pt.y;
    a.z += pt.z;
    a.i += pt.intensity;
    ++a.n;
  }
  out.clear();
  out.reserve(cells.size());
  for (const auto & kv : cells) {
    const auto & a = kv.second;
    out.push_back({static_cast<float>(a.x / a.n), static_cast<float>(a.y / a.n),
                   static_cast<float>(a.z / a.n), static_cast<float>(a.i / a.n)});
  }
}

double now_us() {
  return std::chrono::duration<double, std::micro>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

void report(const char * name, std::vector<double> v) {
  std::sort(v.begin(), v.end());
  double mean = 0;
  for (double x : v) mean += x;
  mean /= static_cast<double>(v.size());
  auto q = [&](double p) {
    return v[std::min(v.size() - 1, static_cast<size_t>(p * static_cast<double>(v.size() - 1) + 0.5))];
  };
  std::printf("  %-34s mean %7.1f  p50 %7.1f  p99 %7.1f  p99.9 %7.1f  max %8.1f  [us]\n", name, mean,
              q(0.5), q(0.99), q(0.999), v.back());
}

volatile double g_sink = 0;  // 結果を使ったことにして、最適化で消されないようにする

// V4(試作): V3 と同じ平らなハッシュ表だが、表の大きさを「点の数」ではなく「前回のボクセルの数」から決める。
// 点が約 33,000 でも、ボクセルは約 11,000 なので、表が 131,072 枠(約 6 MB)でなく 32,768 枠(約 1.5 MB)で済む。
// 満ちすぎたら(負荷率が 0.5 を超えたら)、表を大きくして、やり直す。
struct FlatSmall {
  std::vector<VoxelScratch::Slot> table;
  std::vector<uint32_t> used;
  uint32_t epoch = 0;
  int shift = 0;
  size_t hint = 1024;  // 前回のボクセルの数
  void run(const std::vector<Point> & in, float leaf, std::vector<Point> & out) {
    for (;;) {
      size_t want = 16;
      int bits = 4;
      while (want < 2 * hint + 64) {
        want <<= 1;
        ++bits;
      }
      if (want > table.size()) {
        table.assign(want, VoxelScratch::Slot{});
        shift = 64 - bits;
        epoch = 0;
        used.reserve(in.size());
      }
      ++epoch;
      used.clear();
      const size_t mask = table.size() - 1;
      const size_t limit = table.size() / 2;
      bool overflow = false;
      for (const auto & p : in) {
        const uint64_t k = key_of(p, leaf);
        size_t h = static_cast<size_t>((k * 0x9E3779B97F4A7C15ull) >> shift);
        for (;;) {
          auto & s = table[h];
          if (s.stamp != epoch) {
            s.stamp = epoch;
            s.key = k;
            s.x = p.x;
            s.y = p.y;
            s.z = p.z;
            s.i = p.intensity;
            s.n = 1;
            used.push_back(static_cast<uint32_t>(h));
            break;
          }
          if (s.key == k) {
            s.x += p.x;
            s.y += p.y;
            s.z += p.z;
            s.i += p.intensity;
            ++s.n;
            break;
          }
          h = (h + 1) & mask;
        }
        if (used.size() > limit) {
          overflow = true;
          break;
        }
      }
      if (overflow) {
        hint = used.size() * 2;
        continue;
      }
      out.clear();
      out.reserve(used.size());
      for (uint32_t h : used) {
        const auto & a = table[h];
        out.push_back({static_cast<float>(a.x / a.n), static_cast<float>(a.y / a.n),
                       static_cast<float>(a.z / a.n), static_cast<float>(a.i / a.n)});
      }
      hint = used.size();
      return;
    }
  }
};

// 1 つの条件ごとに別々に、毎回 interval_ms 休んでから測る(0 なら休まない)。
// 休む長さを 100 にすると、パイプライン(10 Hz)と同じ「キャッシュも CPU も冷えた状態」になる。
template <class F>
void run_pass(const char * name, int iters, int interval_ms, F && fn) {
  std::vector<double> v;
  for (int it = -20; it < iters; ++it) {
    if (interval_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    const double t0 = now_us();
    fn(static_cast<size_t>(it + 20));
    const double t1 = now_us();
    if (it >= 0) v.push_back(t1 - t0);
  }
  report(name, v);
}

}  // namespace

int main(int argc, char ** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 3000;
  constexpr int kWarm = 300;
  const float leaf = 0.2f;

  SimParams sp;
  std::vector<Box> cars;
  const float lanes[4] = {-7.f, -3.5f, 3.5f, 7.f};
  for (int i = 0; i < 8; ++i) {
    const float x = -30.f + 8.f * static_cast<float>(i), y = lanes[i % 4];
    cars.push_back({x - 2.25f, y - 0.9f, -1.8f, x + 2.25f, y + 0.9f, -0.3f});
  }
  const Box keep{-50, -50, -2.5f, 50, 50, 1}, ego{-2.5f, -1.2f, -3, 2.5f, 1.2f, 1};
  std::vector<std::vector<Point>> frames;
  std::vector<std::vector<uint8_t>> msgs;  // PointCloud2 の data 相当
  size_t lo = ~size_t(0), hi = 0;
  for (uint32_t f = 0; f < 50; ++f) {
    frames.push_back(crop_box(simulate_scan(sp, cars, f), keep, ego));
    msgs.emplace_back(frames.back().size() * sizeof(Point));
    std::memcpy(msgs.back().data(), frames.back().data(), msgs.back().size());
    lo = std::min(lo, frames.back().size());
    hi = std::max(hi, frames.back().size());
  }
  std::printf("frames=50, points per frame: %zu .. %zu, leaf=%.2f, iterations=%d\n", lo, hi, leaf, iters);

  // 作り方を変えても、結果が同じか(ボクセルごとの重心の差)
  {
    Map cells;
    std::vector<Point> a, b;
    voxel_with_map(frames[0], leaf, cells, a);
    VoxelScratch scratch;
    voxel_downsample(frames[0], leaf, scratch, b);
    std::unordered_map<uint64_t, Point> ref;
    for (const auto & p : a) ref[key_of(p, leaf)] = p;
    double md = 0;
    bool ok = a.size() == b.size();
    for (const auto & q : b) {
      const auto it = ref.find(key_of(q, leaf));
      if (it == ref.end()) {
        ok = false;
        break;
      }
      md = std::max({md, std::fabs(double(it->second.x) - q.x), std::fabs(double(it->second.y) - q.y),
                     std::fabs(double(it->second.z) - q.z)});
    }
    std::printf("check: flat table vs unordered_map: %s (voxels %zu / %zu, max diff %.1e m)\n",
                ok && md < 1e-3 ? "OK" : "MISMATCH", a.size(), b.size(), md);
  }

  if (argc > 2) {  // 条件ごとに別々に測る。alloc_bench 150 100 なら、毎回 100 ms 休む(パイプラインと同じ)
    const int interval_ms = std::atoi(argv[2]);
    std::printf("== 間引き 1 つずつ、毎回 %d ms 休んでから測る\n", interval_ms);
    {
      run_pass("V0 今までの形: 毎回 unordered_map", iters, interval_ms, [&](size_t i) {
        Map fresh;
        std::vector<Point> o;
        voxel_with_map(frames[i % 50], leaf, fresh, o);
        g_sink = g_sink + o.size();
      });
      Map map;
      std::vector<Point> out1;
      run_pass("V1 unordered_map を使い回す", iters, interval_ms, [&](size_t i) {
        voxel_with_map(frames[i % 50], leaf, map, out1);
        g_sink = g_sink + out1.size();
      });
      VoxelScratch scratch;
      std::vector<Point> out3;
      run_pass("V3 平らなハッシュ表を使い回す(今)", iters, interval_ms, [&](size_t i) {
        voxel_downsample(frames[i % 50], leaf, scratch, out3);
        g_sink = g_sink + out3.size();
      });
      FlatSmall small;
      std::vector<Point> out4;
      run_pass("V4 表を小さく(前回のボクセル数から)", iters, interval_ms, [&](size_t i) {
        small.run(frames[i % 50], leaf, out4);
        g_sink = g_sink + out4.size();
      });
    }
    return 0;
  }

  for (int round = 1; round <= 2; ++round) {  // 2 回やって、1 回目と 2 回目が近いことを見る
    std::printf("== round %d\n", round);
    {
      std::vector<double> A, B, C;
      std::vector<Point> reuse;
      for (int it = -kWarm; it < iters; ++it) {
        const auto & m = msgs[static_cast<size_t>(it + kWarm) % 50];
        const size_t n = m.size() / sizeof(Point);
        const double t0 = now_us();
        {
          std::vector<Point> pts(n);
          std::memcpy(pts.data(), m.data(), n * sizeof(Point));
          g_sink = g_sink + pts[n / 2].x;
        }
        const double t1 = now_us();
        {
          std::unique_ptr<Point[]> pts(new Point[n]);
          std::memcpy(pts.get(), m.data(), n * sizeof(Point));
          g_sink = g_sink + pts[n / 2].x;
        }
        const double t2 = now_us();
        {
          reuse.resize(n);
          std::memcpy(reuse.data(), m.data(), n * sizeof(Point));
          g_sink = g_sink + reuse[n / 2].x;
        }
        const double t3 = now_us();
        if (it >= 0) {
          A.push_back(t1 - t0);
          B.push_back(t2 - t1);
          C.push_back(t3 - t2);
        }
      }
      std::printf(" [取り込み] 約 33,000 点(約 530 KB)を vector<Point> に入れる\n");
      report("A 今までの形: vector<Point>(n)", A);
      report("B 0 で埋めない new Point[n]", B);
      report("C 使い回し: resize + memcpy", C);
    }
    {
      std::vector<double> V0, V1, V2, V3;
      Map map;
      std::vector<Point> out1, out3;
      VoxelScratch scratch;
      for (int it = -kWarm; it < iters; ++it) {
        const auto & in = frames[static_cast<size_t>(it + kWarm) % 50];
        const double t0 = now_us();
        {
          Map fresh;
          std::vector<Point> o;
          voxel_with_map(in, leaf, fresh, o);
          g_sink = g_sink + o.size();
        }
        const double t1 = now_us();
        voxel_with_map(in, leaf, map, out1);
        g_sink = g_sink + out1.size();
        const double t2 = now_us();
        {
          const auto o = voxel_downsample(in, leaf);
          g_sink = g_sink + o.size();
        }
        const double t3 = now_us();
        voxel_downsample(in, leaf, scratch, out3);
        g_sink = g_sink + out3.size();
        const double t4 = now_us();
        if (it >= 0) {
          V0.push_back(t1 - t0);
          V1.push_back(t2 - t1);
          V2.push_back(t3 - t2);
          V3.push_back(t4 - t3);
        }
      }
      std::printf(" [間引き] voxel_downsample(約 33,000 点 → 約 11,000 点)\n");
      report("V0 今までの形: 毎回 unordered_map", V0);
      report("V1 unordered_map を使い回す", V1);
      report("V2 平らなハッシュ表(毎回作る)", V2);
      report("V3 平らなハッシュ表を使い回す(今)", V3);
    }
    {
      std::vector<double> X, Y;
      std::vector<Point> reuse, outb;
      VoxelScratch scratch;
      for (int it = -kWarm; it < iters; ++it) {
        const auto & m = msgs[static_cast<size_t>(it + kWarm) % 50];
        const size_t n = m.size() / sizeof(Point);
        const double t0 = now_us();
        {
          std::vector<Point> pts(n);
          std::memcpy(pts.data(), m.data(), n * sizeof(Point));
          Map fresh;
          std::vector<Point> o;
          voxel_with_map(pts, leaf, fresh, o);
          g_sink = g_sink + o.size();
        }
        const double t1 = now_us();
        {
          reuse.resize(n);
          std::memcpy(reuse.data(), m.data(), n * sizeof(Point));
          voxel_downsample(reuse, leaf, scratch, outb);
          g_sink = g_sink + outb.size();
        }
        const double t2 = now_us();
        if (it >= 0) {
          X.push_back(t1 - t0);
          Y.push_back(t2 - t1);
        }
      }
      std::printf(" [1 フレーム分] 取り込み + 間引き\n");
      report("今までの形(A + V0)", X);
      report("使い回し(C + V3。今)", Y);
    }
  }
  return 0;
}
