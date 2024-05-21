/**
 * @file   pin_utilization_map.cpp
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin
 * @date   Dec 2019
 * @brief  Compute the RUDY/RISA map for routing demand.
 *         A routing/pin utilization estimator based on the following two papers
 *         "Fast and Accurate Routing Demand Estimation for efficient
 * Routability-driven Placement", by Peter Spindler, DATE'07 "RISA: Accurate and
 * Efficient Placement Routability Modeling", by Chih-liang Eric Cheng,
 * ICCAD'94
 */

#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

// fill the demand map pin by pin
/// @param atomic_add_op functional object for atomic add 
/// @param buf_map 2D density map in column-major to write
template <typename T, typename AtomicOp>
int pinDemandMapLauncher(const T *node_x, const T *node_y, const T *node_size_x,
                         const T *node_size_y,
                         const T *half_node_size_stretch_x,
                         const T *half_node_size_stretch_y,
                         const T *pin_weights, T xl, T yl, T xh, T yh,
                         int num_bins_x,
                         int num_bins_y, int num_nodes, 
                         int num_threads, AtomicOp atomic_add_op,
                         typename AtomicOp::type* buf_map) {
  T bin_size_x = (xh - xl) / num_bins_x; 
  T bin_size_y = (yh - yl) / num_bins_y; 
  const T inv_bin_size_x = 1.0 / bin_size_x;
  const T inv_bin_size_y = 1.0 / bin_size_y;

  int chunk_size =
      DREAMPLACE_STD_NAMESPACE::max(int(num_nodes / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for (int i = 0; i < num_nodes; ++i) {
    const T node_center_x = node_x[i] + node_size_x[i] / 2;
    const T node_center_y = node_y[i] + node_size_y[i] / 2;

    const T x_min = node_center_x - half_node_size_stretch_x[i];
    const T x_max = node_center_x + half_node_size_stretch_x[i];
    int bin_index_xl = int((x_min - xl) * inv_bin_size_x);
    int bin_index_xh = int((x_max - xl) * inv_bin_size_x) + 1;
    bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
    bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

    const T y_min = node_center_y - half_node_size_stretch_y[i];
    const T y_max = node_center_y + half_node_size_stretch_y[i];
    int bin_index_yl = int((y_min - yl) * inv_bin_size_y);
    int bin_index_yh = int((y_max - yl) * inv_bin_size_y) + 1;
    bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
    bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

    T density = pin_weights[i] /
                (half_node_size_stretch_x[i] * half_node_size_stretch_y[i] * 4);
    for (int x = bin_index_xl; x < bin_index_xh; ++x) {
      for (int y = bin_index_yl; y < bin_index_yh; ++y) {
        T bin_xl = xl + x * bin_size_x;
        T bin_yl = yl + y * bin_size_y;
        T bin_xh = bin_xl + bin_size_x;
        T bin_yh = bin_yl + bin_size_y;
        T overlap = DREAMPLACE_STD_NAMESPACE::max(
                        DREAMPLACE_STD_NAMESPACE::min(x_max, bin_xh) -
                            DREAMPLACE_STD_NAMESPACE::max(x_min, bin_xl),
                        (T)0) *
                    DREAMPLACE_STD_NAMESPACE::max(
                        DREAMPLACE_STD_NAMESPACE::min(y_max, bin_yh) -
                            DREAMPLACE_STD_NAMESPACE::max(y_min, bin_yl),
                        (T)0);
        int index = x * num_bins_y + y;
        atomic_add_op(&buf_map[index], overlap * density);
      }
    }
  }
  return 0;
}

at::Tensor pin_utilization_map_forward(
    at::Tensor pos, at::Tensor node_size_x, at::Tensor node_size_y,
    at::Tensor half_node_size_stretch_x, at::Tensor half_node_size_stretch_y,
    at::Tensor pin_weights, double xl, double yl, double xh, double yh,
    int num_physical_nodes,
    int num_bins_x, int num_bins_y, 
    int deterministic_flag) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  CHECK_FLAT_CPU(node_size_x);
  CHECK_CONTIGUOUS(node_size_x);
  CHECK_FLAT_CPU(node_size_y);
  CHECK_CONTIGUOUS(node_size_y);

  CHECK_FLAT_CPU(half_node_size_stretch_x);
  CHECK_CONTIGUOUS(half_node_size_stretch_x);

  CHECK_FLAT_CPU(half_node_size_stretch_y);
  CHECK_CONTIGUOUS(half_node_size_stretch_y);

  CHECK_FLAT_CPU(pin_weights);
  CHECK_CONTIGUOUS(pin_weights);

  CHECK_FLAT_CPU(node_size_x);
  CHECK_CONTIGUOUS(node_size_x);

  CHECK_FLAT_CPU(node_size_y);
  CHECK_CONTIGUOUS(node_size_y);

  at::Tensor pin_utilization_map =
      at::zeros({num_bins_x, num_bins_y}, pos.options());
  auto num_nodes = pos.numel() / 2;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "pinDemandMapLauncher", [&] {
      if (deterministic_flag) {
          double diearea = (xh - xl) * (yh - yl);
          int integer_bits = DREAMPLACE_STD_NAMESPACE::max((int)ceil(log2(diearea)) + 1, 32);
          int fraction_bits = DREAMPLACE_STD_NAMESPACE::max(64 - integer_bits, 0);
          long scale_factor = (1L << fraction_bits);
          int num_bins = num_bins_x * num_bins_y;

          std::vector<long> buf_map(num_bins, 0);
          AtomicAdd<long> atomic_add_op(scale_factor);

          pinDemandMapLauncher<scalar_t, decltype(atomic_add_op)>(
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(half_node_size_stretch_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(half_node_size_stretch_y, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(pin_weights, scalar_t), xl, yl, xh, yh,
              num_bins_x, num_bins_y, num_physical_nodes,
              at::get_num_threads(),
              atomic_add_op, buf_map.data());

          scaleAdd(DREAMPLACE_TENSOR_DATA_PTR(pin_utilization_map, scalar_t),
                   buf_map.data(), 1.0 / scale_factor, num_bins,
                   at::get_num_threads());
      } else {
          AtomicAdd<scalar_t> atomic_add_op;
          pinDemandMapLauncher<scalar_t, decltype(atomic_add_op)>(
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(half_node_size_stretch_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(half_node_size_stretch_y, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(pin_weights, scalar_t), xl, yl, xh, yh,
              num_bins_x, num_bins_y, num_physical_nodes,
              at::get_num_threads(),
              atomic_add_op, 
              DREAMPLACE_TENSOR_DATA_PTR(pin_utilization_map, scalar_t));
      }
  });

  return pin_utilization_map;
}

template <typename T, typename AtomicOp>
int netFeatLauncher(const T *pin_pos_x, const T *pin_pos_y,
                 const int *netpin_start, const int *flat_netpin,
                 const T bin_size_x, const T bin_size_y,
                 T xl, T yl, T xh, T yh, int num_bins_x, int num_bins_y,
                 int num_nets, int num_threads, AtomicOp atomic_add_op,
                 typename AtomicOp::type* net_feat) {
  const T inv_bin_size_x = 1.0 / bin_size_x;
  const T inv_bin_size_y = 1.0 / bin_size_y;

  int chunk_size =
      DREAMPLACE_STD_NAMESPACE::max(int(num_nets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for (int i = 0; i < num_nets; ++i) {

    T x_max = -std::numeric_limits<T>::max();
    T x_min = std::numeric_limits<T>::max();
    T y_max = -std::numeric_limits<T>::max();
    T y_min = std::numeric_limits<T>::max();

    for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
      int pin_id = flat_netpin[j];
      const T xx = pin_pos_x[pin_id];
      x_max = DREAMPLACE_STD_NAMESPACE::max(xx, x_max);
      x_min = DREAMPLACE_STD_NAMESPACE::min(xx, x_min);
      const T yy = pin_pos_y[pin_id];
      y_max = DREAMPLACE_STD_NAMESPACE::max(yy, y_max);
      y_min = DREAMPLACE_STD_NAMESPACE::min(yy, y_min);
    }

    // compute the bin box that this net will affect
    int max_px = int(x_max * inv_bin_size_x);
    int max_py = int(y_max * inv_bin_size_y);
    int min_px = int(x_min * inv_bin_size_x);
    int min_py = int(y_min * inv_bin_size_y);
    double span_h = x_max - x_min + 1;
    double span_v = y_max - y_min + 1;
    int span_ph = max_px - min_px + 1;
    int span_pv = max_py - min_py + 1;
    net_feat[i * 7] = span_h;
    net_feat[i * 7 + 1] = span_v;
    net_feat[i * 7 + 2] = span_h * span_v;
    net_feat[i * 7 + 3] = span_ph;
    net_feat[i * 7 + 4] = span_pv;
    net_feat[i * 7 + 5] = span_ph * span_pv;
    net_feat[i * 7 + 6] = netpin_start[i + 1] - netpin_start[i];



    // int bin_index_xl = int((x_min - xl) * inv_bin_size_x);
    // int bin_index_xh = int((x_max - xl) * inv_bin_size_x) + 1;
    // bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
    // bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

    // int bin_index_yl = int((y_min - yl) * inv_bin_size_y);
    // int bin_index_yh = int((y_max - yl) * inv_bin_size_y) + 1;
    // bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
    // bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

    // T wt = netWiringDistributionMapWeight<T>(netpin_start[i + 1] -
    //                                          netpin_start[i]);
    // if (net_weights) {
    //   wt *= net_weights[i];
    // }

    // for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
    //   int pin_id = flat_netpin[j];
    //   const T xx = pin_pos_x[pin_id];
    //   const T yy = pin_pos_y[pin_id];

    //   int bin_index_x = int((xx - xl) * inv_bin_size_x);
    //   int bin_index_y = int((yy - yl) * inv_bin_size_y);
      
    //   int index = bin_index_x * num_bins_y + bin_index_y;
    //   atomic_add_op(&horizontal_buf_map[index], wt / (bin_index_xh - bin_index_xl + std::numeric_limits<T>::epsilon()));
    //   atomic_add_op(&vertical_buf_map[index], wt / (bin_index_yh - bin_index_yl + std::numeric_limits<T>::epsilon())); 
    // }
  }
  return 0;
}

at::Tensor net_feat_forward(at::Tensor pin_pos, at::Tensor netpin_start,
                  at::Tensor flat_netpin, 
                  double bin_size_x, double bin_size_y, double xl, double yl,
                  double xh, double yh, int num_bins_x, int num_bins_y,
                  at::Tensor net_feat) {
  CHECK_FLAT_CUDA(pin_pos);
  CHECK_EVEN(pin_pos);
  CHECK_CONTIGUOUS(pin_pos);

  CHECK_FLAT_CUDA(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);

  CHECK_FLAT_CUDA(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pin_pos.numel() / 2;

  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_pos, "netFeatLauncher", [&] {
    AtomicAdd<scalar_t> atomic_add_op;
    netFeatLauncher<scalar_t, decltype(atomic_add_op)>(
        DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t) + num_pins,
        DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
        bin_size_x, bin_size_y, xl, yl, xh, yh,

        num_bins_x, num_bins_y, num_nets, at::get_num_threads(),
        atomic_add_op, 
        DREAMPLACE_TENSOR_DATA_PTR(net_feat, scalar_t));
  });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::pin_utilization_map_forward,
        "compute pin utilization map");
  m.def("feat_forward", &DREAMPLACE_NAMESPACE::net_feat_forward,
        "compute net feat");
}
