/**
 * @file   pin_utilization_map_cuda_kernel.cu
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin
 * @date   Dec 2019
 * @brief  Compute the RUDY/RISA map for routing demand. 
 *         A routing/pin utilization estimator based on the following two papers
 *         "Fast and Accurate Routing Demand Estimation for efficient Routability-driven Placement", by Peter Spindler, DATE'07
 *         "RISA: Accurate and Efficient Placement Routability Modeling", by Chih-liang Eric Cheng, ICCAD'94
 */

#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

// fill the demand map net by net
template <typename T, typename AtomicOp>
__global__ void pinDemandMap(const T *node_x, const T *node_y,
                          const T *node_size_x, const T *node_size_y,
                          const T *half_node_size_stretch_x, const T *half_node_size_stretch_y,
                          const T *pin_weights,
                          T xl, T yl, T xh, T yh,
                          T bin_size_x, T bin_size_y,
                          int num_bins_x, int num_bins_y,
                          int num_nodes, AtomicOp atomic_add_op,
                          typename AtomicOp::type *buf_map
                          )
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < num_nodes)
    {
        const T node_center_x = node_x[i] + node_size_x[i]/2; 
        const T node_center_y = node_y[i] + node_size_y[i]/2; 

        const T x_min = node_center_x - half_node_size_stretch_x[i];
        const T x_max = node_center_x + half_node_size_stretch_x[i];
        int bin_index_xl = int((x_min - xl) / bin_size_x);
        int bin_index_xh = int((x_max - xl) / bin_size_x) + 1;
        bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

        const T y_min = node_center_y - half_node_size_stretch_y[i];
        const T y_max = node_center_y + half_node_size_stretch_y[i];
        int bin_index_yl = int((y_min - yl) / bin_size_y);
        int bin_index_yh = int((y_max - yl) / bin_size_y) + 1;
        bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
        bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

        T density = pin_weights[i] / (half_node_size_stretch_x[i] * half_node_size_stretch_y[i] * 4);
        for (int x = bin_index_xl; x < bin_index_xh; ++x)
        {
            for (int y = bin_index_yl; y < bin_index_yh; ++y)
            {
                T bin_xl = xl + x * bin_size_x; 
                T bin_yl = yl + y * bin_size_y; 
                T bin_xh = bin_xl + bin_size_x; 
                T bin_yh = bin_yl + bin_size_y; 
                T overlap = DREAMPLACE_STD_NAMESPACE::max(DREAMPLACE_STD_NAMESPACE::min(x_max, bin_xh) - DREAMPLACE_STD_NAMESPACE::max(x_min, bin_xl), (T)0) *
                            DREAMPLACE_STD_NAMESPACE::max(DREAMPLACE_STD_NAMESPACE::min(y_max, bin_yh) - DREAMPLACE_STD_NAMESPACE::max(y_min, bin_yl), (T)0);
                int index = x * num_bins_y + y;
                atomic_add_op(&buf_map[index], overlap * density);
            }
        }
    }
}

// fill the demand map net by net
template <typename T>
int pinDemandMapCudaLauncher(const T *node_x, const T *node_y,
                          const T *node_size_x, const T *node_size_y,
                          const T *half_node_size_stretch_x, const T *half_node_size_stretch_y,
                          const T *pin_weights,
                          T xl, T yl, T xh, T yh,
                          int num_bins_x, int num_bins_y,
                          int num_nodes,
                          bool deterministic_flag, 
                          T *pin_utilization_map
                          )
{
  T bin_size_x = (xh - xl) / num_bins_x; 
  T bin_size_y = (yh - yl) / num_bins_y; 

  if (deterministic_flag)  // deterministic implementation using unsigned long
                           // as fixed point number
  {
    // total die area
    double diearea = (xh - xl) * (yh - yl);
    int integer_bits = max((int)ceil(log2(diearea)) + 1, 32);
    int fraction_bits = max(64 - integer_bits, 0);
    unsigned long long int scale_factor = (1UL << fraction_bits);
    int num_bins = num_bins_x * num_bins_y;
    unsigned long long int *buf_map = NULL;
    allocateCUDA(buf_map, num_bins, unsigned long long int);

    AtomicAddCUDA<unsigned long long int> atomic_add_op(scale_factor);

    int thread_count = 512;
    int block_count = ceilDiv(num_bins, thread_count);
    copyScaleArray<<<block_count, thread_count>>>(
        buf_map, pin_utilization_map, scale_factor, num_bins);

    block_count = ceilDiv(num_nodes, thread_count);
    pinDemandMap<<<block_count, thread_count>>>(
            node_x, node_y,
            node_size_x, node_size_y,
            half_node_size_stretch_x, half_node_size_stretch_y,
            pin_weights,
            xl, yl, xh, yh,
            bin_size_x, bin_size_y,
            num_bins_x, num_bins_y,
            num_nodes,
            atomic_add_op, 
            buf_map
        );

    block_count = ceilDiv(num_bins, thread_count);
    copyScaleArray<<<block_count, thread_count>>>(
        pin_utilization_map, buf_map, T(1.0 / scale_factor), num_bins);

    destroyCUDA(buf_map);
  } else {
    AtomicAddCUDA<T> atomic_add_op;
    int thread_count = 512;
    int block_count = ceilDiv(num_nodes, thread_count);
    pinDemandMap<<<block_count, thread_count>>>(
            node_x, node_y,
            node_size_x, node_size_y,
            half_node_size_stretch_x, half_node_size_stretch_y,
            pin_weights,
            xl, yl, xh, yh,
            bin_size_x, bin_size_y,
            num_bins_x, num_bins_y,
            num_nodes,
            atomic_add_op,
            pin_utilization_map
        );
  }

    return 0;
}


template <typename T, typename AtomicOp>
__global__ void netFeat(const T *pin_pos_x,
                              const T *pin_pos_y,
                              const int *netpin_start,
                              const int *flat_netpin,
                              T bin_size_x, T bin_size_y,
                              T xl, T yl, T xh, T yh,

                              int num_bins_x, int num_bins_y,
                              int num_nets, AtomicOp atomic_add_op,
                              typename AtomicOp::type *net_feat)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num_nets)
    {
        const int start = netpin_start[i];
        const int end = netpin_start[i + 1];

        T x_max = -cuda::numeric_limits<T>::max();
        T x_min = cuda::numeric_limits<T>::max();
        T y_max = -cuda::numeric_limits<T>::max();
        T y_min = cuda::numeric_limits<T>::max();

        for (int j = start; j < end; ++j)
        {
            int pin_id = flat_netpin[j];
            const T xx = pin_pos_x[pin_id];
            x_max = DREAMPLACE_STD_NAMESPACE::max(xx, x_max);
            x_min = DREAMPLACE_STD_NAMESPACE::min(xx, x_min);
            const T yy = pin_pos_y[pin_id];
            y_max = DREAMPLACE_STD_NAMESPACE::max(yy, y_max);
            y_min = DREAMPLACE_STD_NAMESPACE::min(yy, y_min);
        }

        int max_px = int(x_max / bin_size_x);
        int max_py = int(y_max / bin_size_y);
        int min_px = int(x_min / bin_size_x);
        int min_py = int(y_min / bin_size_y);
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
        // compute the bin box that this net will affect
        // int bin_index_xl = int((x_min - xl) / bin_size_x);
        // int bin_index_xh = int((x_max - xl) / bin_size_x) + 1;
        // bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        // bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

        // int bin_index_yl = int((y_min - yl) / bin_size_y);
        // int bin_index_yh = int((y_max - yl) / bin_size_y) + 1;
        // bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
        // bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

        // T wt = netWiringDistributionMapWeight<T>(end - start);
        // if (net_weights)
        // {
        //     wt *= net_weights[i];
        // }

        // for (int j = start; j < end; ++j) {
        //     int pin_id = flat_netpin[j];
        //     const T xx = pin_pos_x[pin_id];
        //     const T yy = pin_pos_y[pin_id];

        //     int bin_index_x = int((xx - xl) / bin_size_x);
        //     int bin_index_y = int((yy - yl) / bin_size_y);

            
        //     int index = bin_index_x * num_bins_y + bin_index_y;

        //     atomic_add_op(&horizontal_utilization_map[index], wt / (bin_index_xh - bin_index_xl + cuda::numeric_limits<T>::epsilon()));
        //     atomic_add_op(&vertical_utilization_map[index], wt / (bin_index_yh - bin_index_yl + cuda::numeric_limits<T>::epsilon()));
        // }
    }
}

// fill the demand map net by net
template <typename T>
int netFeatCudaLauncher(const T *pin_pos_x,
                              const T *pin_pos_y,
                              const int *netpin_start,
                              const int *flat_netpin,
                              T bin_size_x, T bin_size_y,
                              T xl, T yl, T xh, T yh,

                              int num_bins_x, int num_bins_y,
                              int num_nets, 
                              T *net_feat)
{
    AtomicAddCUDA<T> atomic_add_op;
    int thread_count = 512;
    int block_count = ceilDiv(num_nets, thread_count);
    netFeat<<<block_count, thread_count>>>(
            pin_pos_x,
            pin_pos_y,
            netpin_start,
            flat_netpin,
            bin_size_x, bin_size_y,
            xl, yl, xh, yh,
            num_bins_x, num_bins_y,
            num_nets,
            atomic_add_op, 
            net_feat
            );
  return 0;
}

#define REGISTER_NET_FEAT_KERNEL_LAUNCHER(T)                                           \
    template int netFeatCudaLauncher<T>(const T *pin_pos_x,             \
                                              const T *pin_pos_y,             \
                                              const int *netpin_start,        \
                                              const int *flat_netpin,         \
                                              T bin_size_x, T bin_size_y,     \
                                              T xl, T yl, T xh, T yh,         \
                                                                              \
                                              int num_bins_x, int num_bins_y, \
                                              int num_nets,                   \
                                              T *net_feat);  \

REGISTER_NET_FEAT_KERNEL_LAUNCHER(float);
REGISTER_NET_FEAT_KERNEL_LAUNCHER(double);

#define REGISTER_KERNEL_LAUNCHER(T)                                                                       \
    template int pinDemandMapCudaLauncher<T>(const T *node_x, const T *node_y, \
            const T *node_size_x, const T *node_size_y, \
            const T *half_node_size_stretch_x, const T *half_node_size_stretch_y, \
            const T *pin_weights, \
            T xl, T yl, T xh, T yh, \
            int num_bins_x, int num_bins_y, \
            int num_nodes, \
            bool deterministic_flag, \
            T *pin_utilization_map \
            );

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
