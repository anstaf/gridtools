

#include "example_2.hpp"

#include <gridtools/stencil_composition/stencil_composition.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <type_traits>

static constexpr int MAX_DIM = 3;

namespace transportdelp____gtx86_d7fde2f45f_pyext {

namespace {

// Backend
using backend_t = gt::backend::x86;

// Storage definitions
template <typename T, int Id, bool... Dims>
struct storage_traits {
  using info_t =
      gt::storage_traits<backend_t>::special_storage_info_t<Id, gt::selector<Dims...>,
                                                            gt::halo<0, 0, 0 /* not used */>>;
  using store_t = gt::storage_traits<backend_t>::data_store_t<T, info_t>;
};

template <typename T, int Id>
struct storage_traits<T, Id, 1, 1, 1> {
  using info_t =
      gt::storage_traits<backend_t>::storage_info_t<Id, 3, gt::halo<0, 0, 0 /* not used */>>;
  using store_t = gt::storage_traits<backend_t>::data_store_t<T, info_t>;
};

template <typename T>
constexpr int count(std::integer_sequence<T>) {
  return 0;
}

template <typename T, T Head, T... Tail>
constexpr int count(std::integer_sequence<T, Head, Tail...>) {
  return Head + count(std::integer_sequence<T, Tail...>());
}

template <class T, T... Ints>
constexpr T get(std::integer_sequence<T, Ints...>, std::size_t i) {
  constexpr T arr[] = {Ints...};
  return arr[i];
}

template <typename T, int Id, int NDIM, bool... Dims>
typename storage_traits<T, Id, Dims...>::store_t
make_data_store(const BufferInfo& bi, const std::array<gt::uint_t, MAX_DIM>& compute_domain_shape,
                const std::array<gt::uint_t, NDIM>& origin, gt::selector<Dims...> mask) {
  // ptr, dims and strides are "outer domain" (i.e., compute domain + halo
  // region). The halo region is only defined through `make_grid` (and
  // currently, in the storage info)
  static_assert(NDIM == count(mask), "Mask dimensions do not match origin");

  gt::array<gt::uint_t, MAX_DIM> dims{};
  gt::array<gt::uint_t, MAX_DIM> strides{};
  T* ptr = static_cast<T*>(bi.ptr);

  int curr_stride = 0;
  for(int i = 0, j = 0; i < MAX_DIM; ++i) {
    if(get(mask, i) != 0) {
      curr_stride = strides[i] = bi.strides[j] / sizeof(T);
      ptr += strides[i] * origin[j];
      dims[i] = compute_domain_shape[i] + 2 * origin[j];
      ++j;
    } else {
      strides[i] = curr_stride;
      dims[i] = 0;
    }
  }

  using storage_info_t = typename storage_traits<T, Id, Dims...>::info_t;
  using data_store_t = typename storage_traits<T, Id, Dims...>::store_t;

  return data_store_t{storage_info_t{dims, strides}, ptr, gt::ownership::external_cpu};
}

// Axis
static constexpr gt::uint_t level_offset_limit = 2;

using axis_t = gt::axis<1, /* NIntervals */
                        gt::axis_config::offset_limit<level_offset_limit>>;

// These halo sizes are used to determine the sizes of the temporaries
static constexpr gt::uint_t halo_size_i = 1;
static constexpr gt::uint_t halo_size_j = 1;
static constexpr gt::uint_t halo_size_k = 0;

// Placeholder definitions

using p_delp = gt::arg<0, typename storage_traits<float64_t, 0, true, true, true>::store_t>;
using p_pt = gt::arg<1, typename storage_traits<float64_t, 1, true, true, true>::store_t>;
using p_utc = gt::arg<2, typename storage_traits<float64_t, 2, true, true, true>::store_t>;
using p_vtc = gt::arg<3, typename storage_traits<float64_t, 3, true, true, true>::store_t>;
using p_w = gt::arg<4, typename storage_traits<float64_t, 4, true, true, true>::store_t>;
using p_rarea = gt::arg<5, typename storage_traits<float64_t, 5, true, true, true>::store_t>;
using p_delpc = gt::arg<6, typename storage_traits<float64_t, 6, true, true, true>::store_t>;
using p_ptc = gt::arg<7, typename storage_traits<float64_t, 7, true, true, true>::store_t>;
using p_wc = gt::arg<8, typename storage_traits<float64_t, 8, true, true, true>::store_t>;

using p_domain_size_I = gt::arg<9, gt::global_parameter<gt::uint_t>>;
using p_domain_size_J = gt::arg<10, gt::global_parameter<gt::uint_t>>;
using p_domain_size_K = gt::arg<11, gt::global_parameter<gt::uint_t>>;

// All temporaries are 3D storages. For now...
using p_q__49c_6_11__0ac_24_15 =
    gt::tmp_arg<12, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_sw_mult__49c_6_11__0ac_24_15 =
    gt::tmp_arg<13, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_se_mult__49c_6_11__0ac_24_15 =
    gt::tmp_arg<14, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_nw_mult__49c_6_11__0ac_24_15 =
    gt::tmp_arg<15, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_ne_mult__49c_6_11__0ac_24_15 =
    gt::tmp_arg<16, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_RETURN_VALUE__49c_6_11__0ac_24_15 =
    gt::tmp_arg<17, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_q__49c_6_11__0ac_25_13 =
    gt::tmp_arg<18, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_sw_mult__49c_6_11__0ac_25_13 =
    gt::tmp_arg<19, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_se_mult__49c_6_11__0ac_25_13 =
    gt::tmp_arg<20, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_nw_mult__49c_6_11__0ac_25_13 =
    gt::tmp_arg<21, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_ne_mult__49c_6_11__0ac_25_13 =
    gt::tmp_arg<22, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_RETURN_VALUE__49c_6_11__0ac_25_13 =
    gt::tmp_arg<23, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_q__49c_6_11__0ac_26_12 =
    gt::tmp_arg<24, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_sw_mult__49c_6_11__0ac_26_12 =
    gt::tmp_arg<25, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_se_mult__49c_6_11__0ac_26_12 =
    gt::tmp_arg<26, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_nw_mult__49c_6_11__0ac_26_12 =
    gt::tmp_arg<27, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_ne_mult__49c_6_11__0ac_26_12 =
    gt::tmp_arg<28, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_RETURN_VALUE__49c_6_11__0ac_26_12 =
    gt::tmp_arg<29, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_fx = gt::tmp_arg<30, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_fx1 = gt::tmp_arg<31, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_fx2 = gt::tmp_arg<32, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_q__568_6_11__f4a_30_15 =
    gt::tmp_arg<33, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_sw_mult__568_6_11__f4a_30_15 =
    gt::tmp_arg<34, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_se_mult__568_6_11__f4a_30_15 =
    gt::tmp_arg<35, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_nw_mult__568_6_11__f4a_30_15 =
    gt::tmp_arg<36, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_ne_mult__568_6_11__f4a_30_15 =
    gt::tmp_arg<37, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_RETURN_VALUE__568_6_11__f4a_30_15 =
    gt::tmp_arg<38, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_q__568_6_11__f4a_31_13 =
    gt::tmp_arg<39, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_sw_mult__568_6_11__f4a_31_13 =
    gt::tmp_arg<40, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_se_mult__568_6_11__f4a_31_13 =
    gt::tmp_arg<41, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_nw_mult__568_6_11__f4a_31_13 =
    gt::tmp_arg<42, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_ne_mult__568_6_11__f4a_31_13 =
    gt::tmp_arg<43, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_RETURN_VALUE__568_6_11__f4a_31_13 =
    gt::tmp_arg<44, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_q__568_6_11__f4a_32_12 =
    gt::tmp_arg<45, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_sw_mult__568_6_11__f4a_32_12 =
    gt::tmp_arg<46, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_se_mult__568_6_11__f4a_32_12 =
    gt::tmp_arg<47, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_nw_mult__568_6_11__f4a_32_12 =
    gt::tmp_arg<48, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_ne_mult__568_6_11__f4a_32_12 =
    gt::tmp_arg<49, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_RETURN_VALUE__568_6_11__f4a_32_12 =
    gt::tmp_arg<50, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_fy = gt::tmp_arg<51, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_fy1 = gt::tmp_arg<52, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;
// All temporaries are 3D storages. For now...
using p_fy2 = gt::tmp_arg<53, typename storage_traits<float64_t, 0, 1, 1, 1>::store_t>;

// Constants

// Functors

struct stage__284_func {
  using delp = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using q__49c_6_11__0ac_24_15 = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<delp, q__49c_6_11__0ac_24_15>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(q__49c_6_11__0ac_24_15()) = eval(delp());
  }
};

struct stage__287_func {
  using sw_mult__49c_6_11__0ac_24_15 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<sw_mult__49c_6_11__0ac_24_15>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(sw_mult__49c_6_11__0ac_24_15()) = float64_t{1.0};
  }
};

struct stage__290_func {
  using se_mult__49c_6_11__0ac_24_15 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<se_mult__49c_6_11__0ac_24_15>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(se_mult__49c_6_11__0ac_24_15()) = float64_t{1.0};
  }
};

struct stage__293_func {
  using nw_mult__49c_6_11__0ac_24_15 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<nw_mult__49c_6_11__0ac_24_15>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(nw_mult__49c_6_11__0ac_24_15()) = float64_t{1.0};
  }
};

struct stage__296_func {
  using ne_mult__49c_6_11__0ac_24_15 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<ne_mult__49c_6_11__0ac_24_15>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(ne_mult__49c_6_11__0ac_24_15()) = float64_t{1.0};
  }
};

struct stage__299_func {
  using sw_mult__49c_6_11__0ac_24_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<0, 0, 0, 1, 0, 0>>;
  using q__49c_6_11__0ac_24_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<sw_mult__49c_6_11__0ac_24_15, delp, q__49c_6_11__0ac_24_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= 0 + 0 && eval.i() < 0 + 1 && eval.j() >= 0 + 0 && eval.j() < 0 + 1) {
      eval(q__49c_6_11__0ac_24_15()) = eval(sw_mult__49c_6_11__0ac_24_15()) * eval(delp(0, 1, 0));
    }
  }
};

struct stage__302_func {
  using sw_mult__49c_6_11__0ac_24_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<0, 1, 0, 2, 0, 0>>;
  using q__49c_6_11__0ac_24_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<sw_mult__49c_6_11__0ac_24_15, delp, q__49c_6_11__0ac_24_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 0 && eval.j() >= 0 + 0 && eval.j() < 0 + 1) {
      eval(q__49c_6_11__0ac_24_15()) = eval(sw_mult__49c_6_11__0ac_24_15()) * eval(delp(1, 2, 0));
    }
  }
};

struct stage__305_func {
  using se_mult__49c_6_11__0ac_24_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<0, 0, 0, 1, 0, 0>>;
  using q__49c_6_11__0ac_24_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<se_mult__49c_6_11__0ac_24_15, delp, q__49c_6_11__0ac_24_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 &&
       eval.i() < static_cast<gt::int_t>(eval(domain_size_I())) + 0 && eval.j() >= 0 + 0 &&
       eval.j() < 0 + 1) {
      eval(q__49c_6_11__0ac_24_15()) = eval(se_mult__49c_6_11__0ac_24_15()) * eval(delp(0, 1, 0));
    }
  }
};

struct stage__308_func {
  using se_mult__49c_6_11__0ac_24_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<-1, 0, 0, 2, 0, 0>>;
  using q__49c_6_11__0ac_24_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<se_mult__49c_6_11__0ac_24_15, delp, q__49c_6_11__0ac_24_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) + 0 && eval.j() >= 0 + 0 &&
       eval.j() < 0 + 1) {
      eval(q__49c_6_11__0ac_24_15()) = eval(se_mult__49c_6_11__0ac_24_15()) * eval(delp(-1, 2, 0));
    }
  }
};

struct stage__311_func {
  using nw_mult__49c_6_11__0ac_24_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<0, 0, -1, 0, 0, 0>>;
  using q__49c_6_11__0ac_24_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<nw_mult__49c_6_11__0ac_24_15, delp, q__49c_6_11__0ac_24_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= 0 + 0 && eval.i() < 0 + 1 &&
       eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__49c_6_11__0ac_24_15()) = eval(nw_mult__49c_6_11__0ac_24_15()) * eval(delp(0, -1, 0));
    }
  }
};

struct stage__314_func {
  using nw_mult__49c_6_11__0ac_24_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<0, 1, -2, 0, 0, 0>>;
  using q__49c_6_11__0ac_24_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<nw_mult__49c_6_11__0ac_24_15, delp, q__49c_6_11__0ac_24_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 0 && eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__49c_6_11__0ac_24_15()) = eval(nw_mult__49c_6_11__0ac_24_15()) * eval(delp(1, -2, 0));
    }
  }
};

struct stage__317_func {
  using ne_mult__49c_6_11__0ac_24_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<0, 0, -1, 0, 0, 0>>;
  using q__49c_6_11__0ac_24_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<ne_mult__49c_6_11__0ac_24_15, delp, q__49c_6_11__0ac_24_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 &&
       eval.i() < static_cast<gt::int_t>(eval(domain_size_I())) + 0 &&
       eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__49c_6_11__0ac_24_15()) = eval(ne_mult__49c_6_11__0ac_24_15()) * eval(delp(0, -1, 0));
    }
  }
};

struct stage__320_func {
  using ne_mult__49c_6_11__0ac_24_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<-1, 0, -2, 0, 0, 0>>;
  using q__49c_6_11__0ac_24_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<ne_mult__49c_6_11__0ac_24_15, delp, q__49c_6_11__0ac_24_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) + 0 &&
       eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__49c_6_11__0ac_24_15()) = eval(ne_mult__49c_6_11__0ac_24_15()) * eval(delp(-1, -2, 0));
    }
  }
};

struct stage__323_func {
  using q__49c_6_11__0ac_24_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using RETURN_VALUE__49c_6_11__0ac_24_15 = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<q__49c_6_11__0ac_24_15, RETURN_VALUE__49c_6_11__0ac_24_15>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(RETURN_VALUE__49c_6_11__0ac_24_15()) = eval(q__49c_6_11__0ac_24_15());
  }
};

struct stage__326_func {
  using RETURN_VALUE__49c_6_11__0ac_24_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using q__49c_6_11__0ac_25_13 = gt::inout_accessor<3, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list =
      gt::make_param_list<RETURN_VALUE__49c_6_11__0ac_24_15, pt, delp, q__49c_6_11__0ac_25_13>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(delp()) = eval(RETURN_VALUE__49c_6_11__0ac_24_15());
    eval(q__49c_6_11__0ac_25_13()) = eval(pt());
  }
};

struct stage__332_func {
  using sw_mult__49c_6_11__0ac_25_13 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<sw_mult__49c_6_11__0ac_25_13>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(sw_mult__49c_6_11__0ac_25_13()) = float64_t{1.0};
  }
};

struct stage__335_func {
  using se_mult__49c_6_11__0ac_25_13 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<se_mult__49c_6_11__0ac_25_13>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(se_mult__49c_6_11__0ac_25_13()) = float64_t{1.0};
  }
};

struct stage__338_func {
  using nw_mult__49c_6_11__0ac_25_13 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<nw_mult__49c_6_11__0ac_25_13>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(nw_mult__49c_6_11__0ac_25_13()) = float64_t{1.0};
  }
};

struct stage__341_func {
  using ne_mult__49c_6_11__0ac_25_13 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<ne_mult__49c_6_11__0ac_25_13>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(ne_mult__49c_6_11__0ac_25_13()) = float64_t{1.0};
  }
};

struct stage__344_func {
  using sw_mult__49c_6_11__0ac_25_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<0, 0, 0, 1, 0, 0>>;
  using q__49c_6_11__0ac_25_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<sw_mult__49c_6_11__0ac_25_13, pt, q__49c_6_11__0ac_25_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= 0 + 0 && eval.i() < 0 + 1 && eval.j() >= 0 + 0 && eval.j() < 0 + 1) {
      eval(q__49c_6_11__0ac_25_13()) = eval(sw_mult__49c_6_11__0ac_25_13()) * eval(pt(0, 1, 0));
    }
  }
};

struct stage__347_func {
  using sw_mult__49c_6_11__0ac_25_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<0, 1, 0, 2, 0, 0>>;
  using q__49c_6_11__0ac_25_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<sw_mult__49c_6_11__0ac_25_13, pt, q__49c_6_11__0ac_25_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 0 && eval.j() >= 0 + 0 && eval.j() < 0 + 1) {
      eval(q__49c_6_11__0ac_25_13()) = eval(sw_mult__49c_6_11__0ac_25_13()) * eval(pt(1, 2, 0));
    }
  }
};

struct stage__350_func {
  using se_mult__49c_6_11__0ac_25_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<0, 0, 0, 1, 0, 0>>;
  using q__49c_6_11__0ac_25_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<se_mult__49c_6_11__0ac_25_13, pt, q__49c_6_11__0ac_25_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 &&
       eval.i() < static_cast<gt::int_t>(eval(domain_size_I())) + 0 && eval.j() >= 0 + 0 &&
       eval.j() < 0 + 1) {
      eval(q__49c_6_11__0ac_25_13()) = eval(se_mult__49c_6_11__0ac_25_13()) * eval(pt(0, 1, 0));
    }
  }
};

struct stage__353_func {
  using se_mult__49c_6_11__0ac_25_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<-1, 0, 0, 2, 0, 0>>;
  using q__49c_6_11__0ac_25_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<se_mult__49c_6_11__0ac_25_13, pt, q__49c_6_11__0ac_25_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) + 0 && eval.j() >= 0 + 0 &&
       eval.j() < 0 + 1) {
      eval(q__49c_6_11__0ac_25_13()) = eval(se_mult__49c_6_11__0ac_25_13()) * eval(pt(-1, 2, 0));
    }
  }
};

struct stage__356_func {
  using nw_mult__49c_6_11__0ac_25_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<0, 0, -1, 0, 0, 0>>;
  using q__49c_6_11__0ac_25_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<nw_mult__49c_6_11__0ac_25_13, pt, q__49c_6_11__0ac_25_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= 0 + 0 && eval.i() < 0 + 1 &&
       eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__49c_6_11__0ac_25_13()) = eval(nw_mult__49c_6_11__0ac_25_13()) * eval(pt(0, -1, 0));
    }
  }
};

struct stage__359_func {
  using nw_mult__49c_6_11__0ac_25_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<0, 1, -2, 0, 0, 0>>;
  using q__49c_6_11__0ac_25_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<nw_mult__49c_6_11__0ac_25_13, pt, q__49c_6_11__0ac_25_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 0 && eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__49c_6_11__0ac_25_13()) = eval(nw_mult__49c_6_11__0ac_25_13()) * eval(pt(1, -2, 0));
    }
  }
};

struct stage__362_func {
  using ne_mult__49c_6_11__0ac_25_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<0, 0, -1, 0, 0, 0>>;
  using q__49c_6_11__0ac_25_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<ne_mult__49c_6_11__0ac_25_13, pt, q__49c_6_11__0ac_25_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 &&
       eval.i() < static_cast<gt::int_t>(eval(domain_size_I())) + 0 &&
       eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__49c_6_11__0ac_25_13()) = eval(ne_mult__49c_6_11__0ac_25_13()) * eval(pt(0, -1, 0));
    }
  }
};

struct stage__365_func {
  using ne_mult__49c_6_11__0ac_25_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<-1, 0, -2, 0, 0, 0>>;
  using q__49c_6_11__0ac_25_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<ne_mult__49c_6_11__0ac_25_13, pt, q__49c_6_11__0ac_25_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) + 0 &&
       eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__49c_6_11__0ac_25_13()) = eval(ne_mult__49c_6_11__0ac_25_13()) * eval(pt(-1, -2, 0));
    }
  }
};

struct stage__368_func {
  using q__49c_6_11__0ac_25_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using RETURN_VALUE__49c_6_11__0ac_25_13 = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<q__49c_6_11__0ac_25_13, RETURN_VALUE__49c_6_11__0ac_25_13>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(RETURN_VALUE__49c_6_11__0ac_25_13()) = eval(q__49c_6_11__0ac_25_13());
  }
};

struct stage__371_func {
  using RETURN_VALUE__49c_6_11__0ac_25_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;
  using q__49c_6_11__0ac_26_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::inout_accessor<3, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list =
      gt::make_param_list<RETURN_VALUE__49c_6_11__0ac_25_13, w, q__49c_6_11__0ac_26_12, pt>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(pt()) = eval(RETURN_VALUE__49c_6_11__0ac_25_13());
    eval(q__49c_6_11__0ac_26_12()) = eval(w());
  }
};

struct stage__377_func {
  using sw_mult__49c_6_11__0ac_26_12 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<sw_mult__49c_6_11__0ac_26_12>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(sw_mult__49c_6_11__0ac_26_12()) = float64_t{1.0};
  }
};

struct stage__380_func {
  using se_mult__49c_6_11__0ac_26_12 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<se_mult__49c_6_11__0ac_26_12>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(se_mult__49c_6_11__0ac_26_12()) = float64_t{1.0};
  }
};

struct stage__383_func {
  using nw_mult__49c_6_11__0ac_26_12 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<nw_mult__49c_6_11__0ac_26_12>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(nw_mult__49c_6_11__0ac_26_12()) = float64_t{1.0};
  }
};

struct stage__386_func {
  using ne_mult__49c_6_11__0ac_26_12 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<ne_mult__49c_6_11__0ac_26_12>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(ne_mult__49c_6_11__0ac_26_12()) = float64_t{1.0};
  }
};

struct stage__389_func {
  using sw_mult__49c_6_11__0ac_26_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<0, 0, 0, 1, 0, 0>>;
  using q__49c_6_11__0ac_26_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<sw_mult__49c_6_11__0ac_26_12, w, q__49c_6_11__0ac_26_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= 0 + 0 && eval.i() < 0 + 1 && eval.j() >= 0 + 0 && eval.j() < 0 + 1) {
      eval(q__49c_6_11__0ac_26_12()) = eval(sw_mult__49c_6_11__0ac_26_12()) * eval(w(0, 1, 0));
    }
  }
};

struct stage__392_func {
  using sw_mult__49c_6_11__0ac_26_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<0, 1, 0, 2, 0, 0>>;
  using q__49c_6_11__0ac_26_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<sw_mult__49c_6_11__0ac_26_12, w, q__49c_6_11__0ac_26_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 0 && eval.j() >= 0 + 0 && eval.j() < 0 + 1) {
      eval(q__49c_6_11__0ac_26_12()) = eval(sw_mult__49c_6_11__0ac_26_12()) * eval(w(1, 2, 0));
    }
  }
};

struct stage__395_func {
  using se_mult__49c_6_11__0ac_26_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<0, 0, 0, 1, 0, 0>>;
  using q__49c_6_11__0ac_26_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<se_mult__49c_6_11__0ac_26_12, w, q__49c_6_11__0ac_26_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 &&
       eval.i() < static_cast<gt::int_t>(eval(domain_size_I())) + 0 && eval.j() >= 0 + 0 &&
       eval.j() < 0 + 1) {
      eval(q__49c_6_11__0ac_26_12()) = eval(se_mult__49c_6_11__0ac_26_12()) * eval(w(0, 1, 0));
    }
  }
};

struct stage__398_func {
  using se_mult__49c_6_11__0ac_26_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<-1, 0, 0, 2, 0, 0>>;
  using q__49c_6_11__0ac_26_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<se_mult__49c_6_11__0ac_26_12, w, q__49c_6_11__0ac_26_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) + 0 && eval.j() >= 0 + 0 &&
       eval.j() < 0 + 1) {
      eval(q__49c_6_11__0ac_26_12()) = eval(se_mult__49c_6_11__0ac_26_12()) * eval(w(-1, 2, 0));
    }
  }
};

struct stage__401_func {
  using nw_mult__49c_6_11__0ac_26_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<0, 0, -1, 0, 0, 0>>;
  using q__49c_6_11__0ac_26_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<nw_mult__49c_6_11__0ac_26_12, w, q__49c_6_11__0ac_26_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= 0 + 0 && eval.i() < 0 + 1 &&
       eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__49c_6_11__0ac_26_12()) = eval(nw_mult__49c_6_11__0ac_26_12()) * eval(w(0, -1, 0));
    }
  }
};

struct stage__404_func {
  using nw_mult__49c_6_11__0ac_26_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<0, 1, -2, 0, 0, 0>>;
  using q__49c_6_11__0ac_26_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<nw_mult__49c_6_11__0ac_26_12, w, q__49c_6_11__0ac_26_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 0 && eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__49c_6_11__0ac_26_12()) = eval(nw_mult__49c_6_11__0ac_26_12()) * eval(w(1, -2, 0));
    }
  }
};

struct stage__407_func {
  using ne_mult__49c_6_11__0ac_26_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<0, 0, -1, 0, 0, 0>>;
  using q__49c_6_11__0ac_26_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<ne_mult__49c_6_11__0ac_26_12, w, q__49c_6_11__0ac_26_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 &&
       eval.i() < static_cast<gt::int_t>(eval(domain_size_I())) + 0 &&
       eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__49c_6_11__0ac_26_12()) = eval(ne_mult__49c_6_11__0ac_26_12()) * eval(w(0, -1, 0));
    }
  }
};

struct stage__410_func {
  using ne_mult__49c_6_11__0ac_26_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<-1, 0, -2, 0, 0, 0>>;
  using q__49c_6_11__0ac_26_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<ne_mult__49c_6_11__0ac_26_12, w, q__49c_6_11__0ac_26_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) + 0 &&
       eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__49c_6_11__0ac_26_12()) = eval(ne_mult__49c_6_11__0ac_26_12()) * eval(w(-1, -2, 0));
    }
  }
};

struct stage__413_func {
  using q__49c_6_11__0ac_26_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using RETURN_VALUE__49c_6_11__0ac_26_12 = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<q__49c_6_11__0ac_26_12, RETURN_VALUE__49c_6_11__0ac_26_12>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(RETURN_VALUE__49c_6_11__0ac_26_12()) = eval(q__49c_6_11__0ac_26_12());
  }
};

struct stage__416_func {
  using RETURN_VALUE__49c_6_11__0ac_26_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<RETURN_VALUE__49c_6_11__0ac_26_12, w>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(w()) = eval(RETURN_VALUE__49c_6_11__0ac_26_12());
  }
};

struct stage__419_func {
  using utc = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<-1, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<2, gt::extent<-1, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<3, gt::extent<-1, 0, 0, 0, 0, 0>>;
  using fx2 = gt::inout_accessor<4, gt::extent<0, 0, 0, 0, 0, 0>>;
  using fx1 = gt::inout_accessor<5, gt::extent<0, 0, 0, 0, 0, 0>>;
  using fx = gt::inout_accessor<6, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<utc, delp, pt, w, fx2, fx1, fx>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    float64_t fx1__2f8_28_23;
    float64_t fx__2f8_28_23;
    float64_t fx2__2f8_28_23;
    fx1__2f8_28_23 = (eval(utc()) > float64_t{0.0}) ? eval(delp(-1, 0, 0)) : eval(delp());
    fx__2f8_28_23 = (eval(utc()) > float64_t{0.0}) ? eval(pt(-1, 0, 0)) : eval(pt());
    fx2__2f8_28_23 = (eval(utc()) > float64_t{0.0}) ? eval(w(-1, 0, 0)) : eval(w());
    fx1__2f8_28_23 = eval(utc()) * fx1__2f8_28_23;
    fx__2f8_28_23 = fx1__2f8_28_23 * fx__2f8_28_23;
    fx2__2f8_28_23 = fx1__2f8_28_23 * fx2__2f8_28_23;
    eval(fx()) = fx__2f8_28_23;
    eval(fx1()) = fx1__2f8_28_23;
    eval(fx2()) = fx2__2f8_28_23;
  }
};

struct stage__446_func {
  using delp = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using q__568_6_11__f4a_30_15 = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<delp, q__568_6_11__f4a_30_15>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(q__568_6_11__f4a_30_15()) = eval(delp());
  }
};

struct stage__449_func {
  using sw_mult__568_6_11__f4a_30_15 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using se_mult__568_6_11__f4a_30_15 = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list =
      gt::make_param_list<sw_mult__568_6_11__f4a_30_15, se_mult__568_6_11__f4a_30_15>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(sw_mult__568_6_11__f4a_30_15()) = float64_t{1.0};
    eval(se_mult__568_6_11__f4a_30_15()) = float64_t{1.0};
  }
};

struct stage__455_func {
  using ne_mult__568_6_11__f4a_30_15 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using nw_mult__568_6_11__f4a_30_15 = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list =
      gt::make_param_list<ne_mult__568_6_11__f4a_30_15, nw_mult__568_6_11__f4a_30_15>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(nw_mult__568_6_11__f4a_30_15()) = float64_t{1.0};
    eval(ne_mult__568_6_11__f4a_30_15()) = float64_t{1.0};
  }
};

struct stage__461_func {
  using sw_mult__568_6_11__f4a_30_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<0, 1, 0, 0, 0, 0>>;
  using q__568_6_11__f4a_30_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<sw_mult__568_6_11__f4a_30_15, delp, q__568_6_11__f4a_30_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 1 && eval.j() >= 0 + 0 && eval.j() < 0 + 1) {
      eval(q__568_6_11__f4a_30_15()) = eval(sw_mult__568_6_11__f4a_30_15()) * eval(delp(1, 0, 0));
    }
  }
};

struct stage__464_func {
  using sw_mult__568_6_11__f4a_30_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<0, 2, 0, 1, 0, 0>>;
  using q__568_6_11__f4a_30_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<sw_mult__568_6_11__f4a_30_15, delp, q__568_6_11__f4a_30_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 1 && eval.j() < 0 + 0) {
      eval(q__568_6_11__f4a_30_15()) = eval(sw_mult__568_6_11__f4a_30_15()) * eval(delp(2, 1, 0));
    }
  }
};

struct stage__467_func {
  using se_mult__568_6_11__f4a_30_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<-1, 0, 0, 0, 0, 0>>;
  using q__568_6_11__f4a_30_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<se_mult__568_6_11__f4a_30_15, delp, q__568_6_11__f4a_30_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 && eval.j() >= 0 + 0 &&
       eval.j() < 0 + 1) {
      eval(q__568_6_11__f4a_30_15()) = eval(se_mult__568_6_11__f4a_30_15()) * eval(delp(-1, 0, 0));
    }
  }
};

struct stage__470_func {
  using se_mult__568_6_11__f4a_30_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<-2, 0, 0, 1, 0, 0>>;
  using q__568_6_11__f4a_30_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<se_mult__568_6_11__f4a_30_15, delp, q__568_6_11__f4a_30_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 && eval.j() < 0 + 0) {
      eval(q__568_6_11__f4a_30_15()) = eval(se_mult__568_6_11__f4a_30_15()) * eval(delp(-2, 1, 0));
    }
  }
};

struct stage__473_func {
  using nw_mult__568_6_11__f4a_30_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<0, 1, 0, 0, 0, 0>>;
  using q__568_6_11__f4a_30_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<nw_mult__568_6_11__f4a_30_15, delp, q__568_6_11__f4a_30_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 1 && eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__568_6_11__f4a_30_15()) = eval(nw_mult__568_6_11__f4a_30_15()) * eval(delp(1, 0, 0));
    }
  }
};

struct stage__476_func {
  using nw_mult__568_6_11__f4a_30_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<0, 2, -1, 0, 0, 0>>;
  using q__568_6_11__f4a_30_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<nw_mult__568_6_11__f4a_30_15, delp, q__568_6_11__f4a_30_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 1 && eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__568_6_11__f4a_30_15()) = eval(nw_mult__568_6_11__f4a_30_15()) * eval(delp(2, -1, 0));
    }
  }
};

struct stage__479_func {
  using ne_mult__568_6_11__f4a_30_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<-1, 0, 0, 0, 0, 0>>;
  using q__568_6_11__f4a_30_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<ne_mult__568_6_11__f4a_30_15, delp, q__568_6_11__f4a_30_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 &&
       eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__568_6_11__f4a_30_15()) = eval(ne_mult__568_6_11__f4a_30_15()) * eval(delp(-1, 0, 0));
    }
  }
};

struct stage__482_func {
  using ne_mult__568_6_11__f4a_30_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<-2, 0, -1, 0, 0, 0>>;
  using q__568_6_11__f4a_30_15 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<ne_mult__568_6_11__f4a_30_15, delp, q__568_6_11__f4a_30_15,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 &&
       eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__568_6_11__f4a_30_15()) = eval(ne_mult__568_6_11__f4a_30_15()) * eval(delp(-2, -1, 0));
    }
  }
};

struct stage__485_func {
  using q__568_6_11__f4a_30_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using RETURN_VALUE__568_6_11__f4a_30_15 = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<q__568_6_11__f4a_30_15, RETURN_VALUE__568_6_11__f4a_30_15>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(RETURN_VALUE__568_6_11__f4a_30_15()) = eval(q__568_6_11__f4a_30_15());
  }
};

struct stage__488_func {
  using RETURN_VALUE__568_6_11__f4a_30_15 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;
  using q__568_6_11__f4a_31_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::inout_accessor<3, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list =
      gt::make_param_list<RETURN_VALUE__568_6_11__f4a_30_15, pt, q__568_6_11__f4a_31_13, delp>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(delp()) = eval(RETURN_VALUE__568_6_11__f4a_30_15());
    eval(q__568_6_11__f4a_31_13()) = eval(pt());
  }
};

struct stage__494_func {
  using se_mult__568_6_11__f4a_31_13 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using sw_mult__568_6_11__f4a_31_13 = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list =
      gt::make_param_list<se_mult__568_6_11__f4a_31_13, sw_mult__568_6_11__f4a_31_13>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(sw_mult__568_6_11__f4a_31_13()) = float64_t{1.0};
    eval(se_mult__568_6_11__f4a_31_13()) = float64_t{1.0};
  }
};

struct stage__500_func {
  using nw_mult__568_6_11__f4a_31_13 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using ne_mult__568_6_11__f4a_31_13 = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list =
      gt::make_param_list<nw_mult__568_6_11__f4a_31_13, ne_mult__568_6_11__f4a_31_13>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(nw_mult__568_6_11__f4a_31_13()) = float64_t{1.0};
    eval(ne_mult__568_6_11__f4a_31_13()) = float64_t{1.0};
  }
};

struct stage__506_func {
  using sw_mult__568_6_11__f4a_31_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<0, 1, 0, 0, 0, 0>>;
  using q__568_6_11__f4a_31_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<sw_mult__568_6_11__f4a_31_13, pt, q__568_6_11__f4a_31_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 1 && eval.j() >= 0 + 0 && eval.j() < 0 + 1) {
      eval(q__568_6_11__f4a_31_13()) = eval(sw_mult__568_6_11__f4a_31_13()) * eval(pt(1, 0, 0));
    }
  }
};

struct stage__509_func {
  using sw_mult__568_6_11__f4a_31_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<0, 2, 0, 1, 0, 0>>;
  using q__568_6_11__f4a_31_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<sw_mult__568_6_11__f4a_31_13, pt, q__568_6_11__f4a_31_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 1 && eval.j() < 0 + 0) {
      eval(q__568_6_11__f4a_31_13()) = eval(sw_mult__568_6_11__f4a_31_13()) * eval(pt(2, 1, 0));
    }
  }
};

struct stage__512_func {
  using se_mult__568_6_11__f4a_31_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<-1, 0, 0, 0, 0, 0>>;
  using q__568_6_11__f4a_31_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<se_mult__568_6_11__f4a_31_13, pt, q__568_6_11__f4a_31_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 && eval.j() >= 0 + 0 &&
       eval.j() < 0 + 1) {
      eval(q__568_6_11__f4a_31_13()) = eval(se_mult__568_6_11__f4a_31_13()) * eval(pt(-1, 0, 0));
    }
  }
};

struct stage__515_func {
  using se_mult__568_6_11__f4a_31_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<-2, 0, 0, 1, 0, 0>>;
  using q__568_6_11__f4a_31_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<se_mult__568_6_11__f4a_31_13, pt, q__568_6_11__f4a_31_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 && eval.j() < 0 + 0) {
      eval(q__568_6_11__f4a_31_13()) = eval(se_mult__568_6_11__f4a_31_13()) * eval(pt(-2, 1, 0));
    }
  }
};

struct stage__518_func {
  using nw_mult__568_6_11__f4a_31_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<0, 1, 0, 0, 0, 0>>;
  using q__568_6_11__f4a_31_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<nw_mult__568_6_11__f4a_31_13, pt, q__568_6_11__f4a_31_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 1 && eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__568_6_11__f4a_31_13()) = eval(nw_mult__568_6_11__f4a_31_13()) * eval(pt(1, 0, 0));
    }
  }
};

struct stage__521_func {
  using nw_mult__568_6_11__f4a_31_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<0, 2, -1, 0, 0, 0>>;
  using q__568_6_11__f4a_31_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<nw_mult__568_6_11__f4a_31_13, pt, q__568_6_11__f4a_31_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 1 && eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__568_6_11__f4a_31_13()) = eval(nw_mult__568_6_11__f4a_31_13()) * eval(pt(2, -1, 0));
    }
  }
};

struct stage__524_func {
  using ne_mult__568_6_11__f4a_31_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<-1, 0, 0, 0, 0, 0>>;
  using q__568_6_11__f4a_31_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<ne_mult__568_6_11__f4a_31_13, pt, q__568_6_11__f4a_31_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 &&
       eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__568_6_11__f4a_31_13()) = eval(ne_mult__568_6_11__f4a_31_13()) * eval(pt(-1, 0, 0));
    }
  }
};

struct stage__527_func {
  using ne_mult__568_6_11__f4a_31_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<1, gt::extent<-2, 0, -1, 0, 0, 0>>;
  using q__568_6_11__f4a_31_13 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<ne_mult__568_6_11__f4a_31_13, pt, q__568_6_11__f4a_31_13,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 &&
       eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__568_6_11__f4a_31_13()) = eval(ne_mult__568_6_11__f4a_31_13()) * eval(pt(-2, -1, 0));
    }
  }
};

struct stage__530_func {
  using q__568_6_11__f4a_31_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using RETURN_VALUE__568_6_11__f4a_31_13 = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<q__568_6_11__f4a_31_13, RETURN_VALUE__568_6_11__f4a_31_13>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(RETURN_VALUE__568_6_11__f4a_31_13()) = eval(q__568_6_11__f4a_31_13());
  }
};

struct stage__533_func {
  using RETURN_VALUE__568_6_11__f4a_31_13 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using q__568_6_11__f4a_32_12 = gt::inout_accessor<3, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list =
      gt::make_param_list<RETURN_VALUE__568_6_11__f4a_31_13, w, pt, q__568_6_11__f4a_32_12>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(pt()) = eval(RETURN_VALUE__568_6_11__f4a_31_13());
    eval(q__568_6_11__f4a_32_12()) = eval(w());
  }
};

struct stage__539_func {
  using sw_mult__568_6_11__f4a_32_12 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using se_mult__568_6_11__f4a_32_12 = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list =
      gt::make_param_list<sw_mult__568_6_11__f4a_32_12, se_mult__568_6_11__f4a_32_12>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(sw_mult__568_6_11__f4a_32_12()) = float64_t{1.0};
    eval(se_mult__568_6_11__f4a_32_12()) = float64_t{1.0};
  }
};

struct stage__545_func {
  using nw_mult__568_6_11__f4a_32_12 = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using ne_mult__568_6_11__f4a_32_12 = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list =
      gt::make_param_list<nw_mult__568_6_11__f4a_32_12, ne_mult__568_6_11__f4a_32_12>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(nw_mult__568_6_11__f4a_32_12()) = float64_t{1.0};
    eval(ne_mult__568_6_11__f4a_32_12()) = float64_t{1.0};
  }
};

struct stage__551_func {
  using sw_mult__568_6_11__f4a_32_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<0, 1, 0, 0, 0, 0>>;
  using q__568_6_11__f4a_32_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<sw_mult__568_6_11__f4a_32_12, w, q__568_6_11__f4a_32_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 1 && eval.j() >= 0 + 0 && eval.j() < 0 + 1) {
      eval(q__568_6_11__f4a_32_12()) = eval(sw_mult__568_6_11__f4a_32_12()) * eval(w(1, 0, 0));
    }
  }
};

struct stage__554_func {
  using sw_mult__568_6_11__f4a_32_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<0, 2, 0, 1, 0, 0>>;
  using q__568_6_11__f4a_32_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<sw_mult__568_6_11__f4a_32_12, w, q__568_6_11__f4a_32_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 1 && eval.j() < 0 + 0) {
      eval(q__568_6_11__f4a_32_12()) = eval(sw_mult__568_6_11__f4a_32_12()) * eval(w(2, 1, 0));
    }
  }
};

struct stage__557_func {
  using se_mult__568_6_11__f4a_32_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<-1, 0, 0, 0, 0, 0>>;
  using q__568_6_11__f4a_32_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<se_mult__568_6_11__f4a_32_12, w, q__568_6_11__f4a_32_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 && eval.j() >= 0 + 0 &&
       eval.j() < 0 + 1) {
      eval(q__568_6_11__f4a_32_12()) = eval(se_mult__568_6_11__f4a_32_12()) * eval(w(-1, 0, 0));
    }
  }
};

struct stage__560_func {
  using se_mult__568_6_11__f4a_32_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<-2, 0, 0, 1, 0, 0>>;
  using q__568_6_11__f4a_32_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<se_mult__568_6_11__f4a_32_12, w, q__568_6_11__f4a_32_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 && eval.j() < 0 + 0) {
      eval(q__568_6_11__f4a_32_12()) = eval(se_mult__568_6_11__f4a_32_12()) * eval(w(-2, 1, 0));
    }
  }
};

struct stage__563_func {
  using nw_mult__568_6_11__f4a_32_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<0, 1, 0, 0, 0, 0>>;
  using q__568_6_11__f4a_32_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<nw_mult__568_6_11__f4a_32_12, w, q__568_6_11__f4a_32_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 1 && eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__568_6_11__f4a_32_12()) = eval(nw_mult__568_6_11__f4a_32_12()) * eval(w(1, 0, 0));
    }
  }
};

struct stage__566_func {
  using nw_mult__568_6_11__f4a_32_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<0, 2, -1, 0, 0, 0>>;
  using q__568_6_11__f4a_32_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<nw_mult__568_6_11__f4a_32_12, w, q__568_6_11__f4a_32_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() < 0 + 1 && eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__568_6_11__f4a_32_12()) = eval(nw_mult__568_6_11__f4a_32_12()) * eval(w(2, -1, 0));
    }
  }
};

struct stage__569_func {
  using ne_mult__568_6_11__f4a_32_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<-1, 0, 0, 0, 0, 0>>;
  using q__568_6_11__f4a_32_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<ne_mult__568_6_11__f4a_32_12, w, q__568_6_11__f4a_32_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 &&
       eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) - 1 &&
       eval.j() < static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__568_6_11__f4a_32_12()) = eval(ne_mult__568_6_11__f4a_32_12()) * eval(w(-1, 0, 0));
    }
  }
};

struct stage__572_func {
  using ne_mult__568_6_11__f4a_32_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<1, gt::extent<-2, 0, -1, 0, 0, 0>>;
  using q__568_6_11__f4a_32_12 = gt::inout_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;
  using domain_size_I = gt::in_accessor<3>;
  using domain_size_J = gt::in_accessor<4>;
  using domain_size_K = gt::in_accessor<5>;

  using param_list = gt::make_param_list<ne_mult__568_6_11__f4a_32_12, w, q__568_6_11__f4a_32_12,
                                         domain_size_I, domain_size_J, domain_size_K>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    if(eval.i() >= static_cast<gt::int_t>(eval(domain_size_I())) - 1 &&
       eval.j() >= static_cast<gt::int_t>(eval(domain_size_J())) + 0) {
      eval(q__568_6_11__f4a_32_12()) = eval(ne_mult__568_6_11__f4a_32_12()) * eval(w(-2, -1, 0));
    }
  }
};

struct stage__575_func {
  using q__568_6_11__f4a_32_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using RETURN_VALUE__568_6_11__f4a_32_12 = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<q__568_6_11__f4a_32_12, RETURN_VALUE__568_6_11__f4a_32_12>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(RETURN_VALUE__568_6_11__f4a_32_12()) = eval(q__568_6_11__f4a_32_12());
  }
};

struct stage__578_func {
  using RETURN_VALUE__568_6_11__f4a_32_12 = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::inout_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<RETURN_VALUE__568_6_11__f4a_32_12, w>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(w()) = eval(RETURN_VALUE__568_6_11__f4a_32_12());
  }
};

struct stage__581_func {
  using vtc = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using delp = gt::in_accessor<1, gt::extent<0, 0, -1, 0, 0, 0>>;
  using pt = gt::in_accessor<2, gt::extent<0, 0, -1, 0, 0, 0>>;
  using w = gt::in_accessor<3, gt::extent<0, 0, -1, 0, 0, 0>>;
  using fy1 = gt::inout_accessor<4, gt::extent<0, 0, 0, 0, 0, 0>>;
  using fy2 = gt::inout_accessor<5, gt::extent<0, 0, 0, 0, 0, 0>>;
  using fy = gt::inout_accessor<6, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<vtc, delp, pt, w, fy1, fy2, fy>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    float64_t fy1__e76_34_23;
    float64_t fy__e76_34_23;
    float64_t fy2__e76_34_23;
    fy1__e76_34_23 = (eval(vtc()) > float64_t{0.0}) ? eval(delp(0, -1, 0)) : eval(delp());
    fy__e76_34_23 = (eval(vtc()) > float64_t{0.0}) ? eval(pt(0, -1, 0)) : eval(pt());
    fy2__e76_34_23 = (eval(vtc()) > float64_t{0.0}) ? eval(w(0, -1, 0)) : eval(w());
    fy1__e76_34_23 = eval(vtc()) * fy1__e76_34_23;
    fy__e76_34_23 = fy1__e76_34_23 * fy__e76_34_23;
    fy2__e76_34_23 = fy1__e76_34_23 * fy2__e76_34_23;
    eval(fy()) = fy__e76_34_23;
    eval(fy1()) = fy1__e76_34_23;
    eval(fy2()) = fy2__e76_34_23;
  }
};

struct stage__608_func {
  using delp = gt::in_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using fx1 = gt::in_accessor<1, gt::extent<0, 1, 0, 0, 0, 0>>;
  using fy1 = gt::in_accessor<2, gt::extent<0, 0, 0, 1, 0, 0>>;
  using rarea = gt::in_accessor<3, gt::extent<0, 0, 0, 0, 0, 0>>;
  using pt = gt::in_accessor<4, gt::extent<0, 0, 0, 0, 0, 0>>;
  using fx = gt::in_accessor<5, gt::extent<0, 1, 0, 0, 0, 0>>;
  using fy = gt::in_accessor<6, gt::extent<0, 0, 0, 1, 0, 0>>;
  using delpc = gt::inout_accessor<7, gt::extent<0, 0, 0, 0, 0, 0>>;
  using w = gt::in_accessor<8, gt::extent<0, 0, 0, 0, 0, 0>>;
  using fx2 = gt::in_accessor<9, gt::extent<0, 1, 0, 0, 0, 0>>;
  using fy2 = gt::in_accessor<10, gt::extent<0, 0, 0, 1, 0, 0>>;
  using wc = gt::inout_accessor<11, gt::extent<0, 0, 0, 0, 0, 0>>;
  using ptc = gt::inout_accessor<12, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list =
      gt::make_param_list<delp, fx1, fy1, rarea, pt, fx, fy, delpc, w, fx2, fy2, wc, ptc>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(delpc()) =
        eval(delp()) +
        ((((eval(fx1()) - eval(fx1(1, 0, 0))) + eval(fy1())) - eval(fy1(0, 1, 0))) * eval(rarea()));
    eval(ptc()) =
        ((eval(pt()) * eval(delp())) +
         ((((eval(fx()) - eval(fx(1, 0, 0))) + eval(fy())) - eval(fy(0, 1, 0))) * eval(rarea()))) /
        eval(delpc());
    eval(wc()) = ((eval(w()) * eval(delp())) +
                  ((((eval(fx2()) - eval(fx2(1, 0, 0))) + eval(fy2())) - eval(fy2(0, 1, 0))) *
                   eval(rarea()))) /
                 eval(delpc());
  }
};

// Grids and halos
gt::halo_descriptor make_halo_descriptor(gt::uint_t compute_domain_shape) {
  return {0, 0, 0, compute_domain_shape - 1, compute_domain_shape};
}

auto make_grid(const std::array<gt::uint_t, MAX_DIM>& compute_domain_shape) {
  return gt::make_grid(make_halo_descriptor(compute_domain_shape[0]),
                       make_halo_descriptor(compute_domain_shape[1]),
                       axis_t(compute_domain_shape[2]));
}

} // namespace

// Run actual computation
void run(const std::array<gt::uint_t, MAX_DIM>& domain, const BufferInfo& bi_delp,
         const std::array<gt::uint_t, 3>& delp_origin, const BufferInfo& bi_pt,
         const std::array<gt::uint_t, 3>& pt_origin, const BufferInfo& bi_utc,
         const std::array<gt::uint_t, 3>& utc_origin, const BufferInfo& bi_vtc,
         const std::array<gt::uint_t, 3>& vtc_origin, const BufferInfo& bi_w,
         const std::array<gt::uint_t, 3>& w_origin, const BufferInfo& bi_rarea,
         const std::array<gt::uint_t, 3>& rarea_origin, const BufferInfo& bi_delpc,
         const std::array<gt::uint_t, 3>& delpc_origin, const BufferInfo& bi_ptc,
         const std::array<gt::uint_t, 3>& ptc_origin, const BufferInfo& bi_wc,
         const std::array<gt::uint_t, 3>& wc_origin) {
  // Initialize data stores from input buffers
  auto ds_delp = make_data_store<float64_t, 0, 3>(bi_delp, domain, delp_origin,
                                                  gt::selector<true, true, true>{});
  auto ds_pt =
      make_data_store<float64_t, 1, 3>(bi_pt, domain, pt_origin, gt::selector<true, true, true>{});
  auto ds_utc = make_data_store<float64_t, 2, 3>(bi_utc, domain, utc_origin,
                                                 gt::selector<true, true, true>{});
  auto ds_vtc = make_data_store<float64_t, 3, 3>(bi_vtc, domain, vtc_origin,
                                                 gt::selector<true, true, true>{});
  auto ds_w =
      make_data_store<float64_t, 4, 3>(bi_w, domain, w_origin, gt::selector<true, true, true>{});
  auto ds_rarea = make_data_store<float64_t, 5, 3>(bi_rarea, domain, rarea_origin,
                                                   gt::selector<true, true, true>{});
  auto ds_delpc = make_data_store<float64_t, 6, 3>(bi_delpc, domain, delpc_origin,
                                                   gt::selector<true, true, true>{});
  auto ds_ptc = make_data_store<float64_t, 7, 3>(bi_ptc, domain, ptc_origin,
                                                 gt::selector<true, true, true>{});
  auto ds_wc =
      make_data_store<float64_t, 8, 3>(bi_wc, domain, wc_origin, gt::selector<true, true, true>{});

  // Run computation and wait for the synchronization of the output stores
  auto gt_computation = gt::make_positional_computation<backend_t>(
      make_grid(domain), p_domain_size_I() = gt::make_global_parameter(domain[0]),
      p_domain_size_J() = gt::make_global_parameter(domain[1]),
      p_domain_size_K() = gt::make_global_parameter(domain[2]),
      gt::make_multistage(
          gt::execute::parallel(),
          gt::make_stage_with_extent<stage__284_func, gt::extent<-1, 1, -1, 1>>(
              p_delp(), p_q__49c_6_11__0ac_24_15()),
          gt::make_stage_with_extent<stage__287_func, gt::extent<-1, 0, 0, 0>>(
              p_sw_mult__49c_6_11__0ac_24_15()),
          gt::make_stage_with_extent<stage__290_func, gt::extent<0, 1, 0, 0>>(
              p_se_mult__49c_6_11__0ac_24_15()),
          gt::make_stage_with_extent<stage__293_func, gt::extent<-1, 0, 0, 0>>(
              p_nw_mult__49c_6_11__0ac_24_15()),
          gt::make_stage_with_extent<stage__296_func, gt::extent<0, 1, 0, 0>>(
              p_ne_mult__49c_6_11__0ac_24_15()),
          gt::make_stage_with_extent<stage__299_func, gt::extent<-1, 1, -1, 1>>(
              p_sw_mult__49c_6_11__0ac_24_15(), p_delp(), p_q__49c_6_11__0ac_24_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__302_func, gt::extent<-1, 1, -1, 1>>(
              p_sw_mult__49c_6_11__0ac_24_15(), p_delp(), p_q__49c_6_11__0ac_24_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__305_func, gt::extent<-1, 1, -1, 1>>(
              p_se_mult__49c_6_11__0ac_24_15(), p_delp(), p_q__49c_6_11__0ac_24_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__308_func, gt::extent<-1, 1, -1, 1>>(
              p_se_mult__49c_6_11__0ac_24_15(), p_delp(), p_q__49c_6_11__0ac_24_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__311_func, gt::extent<-1, 1, -1, 1>>(
              p_nw_mult__49c_6_11__0ac_24_15(), p_delp(), p_q__49c_6_11__0ac_24_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__314_func, gt::extent<-1, 1, -1, 1>>(
              p_nw_mult__49c_6_11__0ac_24_15(), p_delp(), p_q__49c_6_11__0ac_24_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__317_func, gt::extent<-1, 1, -1, 1>>(
              p_ne_mult__49c_6_11__0ac_24_15(), p_delp(), p_q__49c_6_11__0ac_24_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__320_func, gt::extent<-1, 1, -1, 1>>(
              p_ne_mult__49c_6_11__0ac_24_15(), p_delp(), p_q__49c_6_11__0ac_24_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__323_func, gt::extent<-1, 1, -1, 1>>(
              p_q__49c_6_11__0ac_24_15(), p_RETURN_VALUE__49c_6_11__0ac_24_15())),
      gt::make_multistage(
          gt::execute::parallel(),
          gt::make_stage_with_extent<stage__326_func, gt::extent<-1, 1, -1, 1>>(
              p_RETURN_VALUE__49c_6_11__0ac_24_15(), p_pt(), p_delp(), p_q__49c_6_11__0ac_25_13()),
          gt::make_stage_with_extent<stage__332_func, gt::extent<-1, 0, 0, 0>>(
              p_sw_mult__49c_6_11__0ac_25_13()),
          gt::make_stage_with_extent<stage__335_func, gt::extent<0, 1, 0, 0>>(
              p_se_mult__49c_6_11__0ac_25_13()),
          gt::make_stage_with_extent<stage__338_func, gt::extent<-1, 0, 0, 0>>(
              p_nw_mult__49c_6_11__0ac_25_13()),
          gt::make_stage_with_extent<stage__341_func, gt::extent<0, 1, 0, 0>>(
              p_ne_mult__49c_6_11__0ac_25_13()),
          gt::make_stage_with_extent<stage__344_func, gt::extent<-1, 1, -1, 1>>(
              p_sw_mult__49c_6_11__0ac_25_13(), p_pt(), p_q__49c_6_11__0ac_25_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__347_func, gt::extent<-1, 1, -1, 1>>(
              p_sw_mult__49c_6_11__0ac_25_13(), p_pt(), p_q__49c_6_11__0ac_25_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__350_func, gt::extent<-1, 1, -1, 1>>(
              p_se_mult__49c_6_11__0ac_25_13(), p_pt(), p_q__49c_6_11__0ac_25_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__353_func, gt::extent<-1, 1, -1, 1>>(
              p_se_mult__49c_6_11__0ac_25_13(), p_pt(), p_q__49c_6_11__0ac_25_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__356_func, gt::extent<-1, 1, -1, 1>>(
              p_nw_mult__49c_6_11__0ac_25_13(), p_pt(), p_q__49c_6_11__0ac_25_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__359_func, gt::extent<-1, 1, -1, 1>>(
              p_nw_mult__49c_6_11__0ac_25_13(), p_pt(), p_q__49c_6_11__0ac_25_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__362_func, gt::extent<-1, 1, -1, 1>>(
              p_ne_mult__49c_6_11__0ac_25_13(), p_pt(), p_q__49c_6_11__0ac_25_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__365_func, gt::extent<-1, 1, -1, 1>>(
              p_ne_mult__49c_6_11__0ac_25_13(), p_pt(), p_q__49c_6_11__0ac_25_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__368_func, gt::extent<-1, 1, -1, 1>>(
              p_q__49c_6_11__0ac_25_13(), p_RETURN_VALUE__49c_6_11__0ac_25_13())),
      gt::make_multistage(
          gt::execute::parallel(),
          gt::make_stage_with_extent<stage__371_func, gt::extent<-1, 1, -1, 1>>(
              p_RETURN_VALUE__49c_6_11__0ac_25_13(), p_w(), p_q__49c_6_11__0ac_26_12(), p_pt()),
          gt::make_stage_with_extent<stage__377_func, gt::extent<-1, 0, 0, 0>>(
              p_sw_mult__49c_6_11__0ac_26_12()),
          gt::make_stage_with_extent<stage__380_func, gt::extent<0, 1, 0, 0>>(
              p_se_mult__49c_6_11__0ac_26_12()),
          gt::make_stage_with_extent<stage__383_func, gt::extent<-1, 0, 0, 0>>(
              p_nw_mult__49c_6_11__0ac_26_12()),
          gt::make_stage_with_extent<stage__386_func, gt::extent<0, 1, 0, 0>>(
              p_ne_mult__49c_6_11__0ac_26_12()),
          gt::make_stage_with_extent<stage__389_func, gt::extent<-1, 1, -1, 1>>(
              p_sw_mult__49c_6_11__0ac_26_12(), p_w(), p_q__49c_6_11__0ac_26_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__392_func, gt::extent<-1, 1, -1, 1>>(
              p_sw_mult__49c_6_11__0ac_26_12(), p_w(), p_q__49c_6_11__0ac_26_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__395_func, gt::extent<-1, 1, -1, 1>>(
              p_se_mult__49c_6_11__0ac_26_12(), p_w(), p_q__49c_6_11__0ac_26_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__398_func, gt::extent<-1, 1, -1, 1>>(
              p_se_mult__49c_6_11__0ac_26_12(), p_w(), p_q__49c_6_11__0ac_26_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__401_func, gt::extent<-1, 1, -1, 1>>(
              p_nw_mult__49c_6_11__0ac_26_12(), p_w(), p_q__49c_6_11__0ac_26_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__404_func, gt::extent<-1, 1, -1, 1>>(
              p_nw_mult__49c_6_11__0ac_26_12(), p_w(), p_q__49c_6_11__0ac_26_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__407_func, gt::extent<-1, 1, -1, 1>>(
              p_ne_mult__49c_6_11__0ac_26_12(), p_w(), p_q__49c_6_11__0ac_26_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__410_func, gt::extent<-1, 1, -1, 1>>(
              p_ne_mult__49c_6_11__0ac_26_12(), p_w(), p_q__49c_6_11__0ac_26_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__413_func, gt::extent<-1, 1, -1, 1>>(
              p_q__49c_6_11__0ac_26_12(), p_RETURN_VALUE__49c_6_11__0ac_26_12())),
      gt::make_multistage(
          gt::execute::parallel(),
          gt::make_stage_with_extent<stage__416_func, gt::extent<-1, 1, -1, 1>>(
              p_RETURN_VALUE__49c_6_11__0ac_26_12(), p_w()),
          gt::make_stage_with_extent<stage__419_func, gt::extent<0, 1, 0, 0>>(
              p_utc(), p_delp(), p_pt(), p_w(), p_fx2(), p_fx1(), p_fx()),
          gt::make_stage_with_extent<stage__446_func, gt::extent<0, 0, -1, 1>>(
              p_delp(), p_q__568_6_11__f4a_30_15()),
          gt::make_stage_with_extent<stage__449_func, gt::extent<0, 0, -1, 0>>(
              p_sw_mult__568_6_11__f4a_30_15(), p_se_mult__568_6_11__f4a_30_15()),
          gt::make_stage_with_extent<stage__455_func, gt::extent<0, 0, 0, 1>>(
              p_ne_mult__568_6_11__f4a_30_15(), p_nw_mult__568_6_11__f4a_30_15()),
          gt::make_stage_with_extent<stage__461_func, gt::extent<0, 0, -1, 1>>(
              p_sw_mult__568_6_11__f4a_30_15(), p_delp(), p_q__568_6_11__f4a_30_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__464_func, gt::extent<0, 0, -1, 1>>(
              p_sw_mult__568_6_11__f4a_30_15(), p_delp(), p_q__568_6_11__f4a_30_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__467_func, gt::extent<0, 0, -1, 1>>(
              p_se_mult__568_6_11__f4a_30_15(), p_delp(), p_q__568_6_11__f4a_30_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__470_func, gt::extent<0, 0, -1, 1>>(
              p_se_mult__568_6_11__f4a_30_15(), p_delp(), p_q__568_6_11__f4a_30_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__473_func, gt::extent<0, 0, -1, 1>>(
              p_nw_mult__568_6_11__f4a_30_15(), p_delp(), p_q__568_6_11__f4a_30_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__476_func, gt::extent<0, 0, -1, 1>>(
              p_nw_mult__568_6_11__f4a_30_15(), p_delp(), p_q__568_6_11__f4a_30_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__479_func, gt::extent<0, 0, -1, 1>>(
              p_ne_mult__568_6_11__f4a_30_15(), p_delp(), p_q__568_6_11__f4a_30_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__482_func, gt::extent<0, 0, -1, 1>>(
              p_ne_mult__568_6_11__f4a_30_15(), p_delp(), p_q__568_6_11__f4a_30_15(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__485_func, gt::extent<0, 0, -1, 1>>(
              p_q__568_6_11__f4a_30_15(), p_RETURN_VALUE__568_6_11__f4a_30_15())),
      gt::make_multistage(
          gt::execute::parallel(),
          gt::make_stage_with_extent<stage__488_func, gt::extent<0, 0, -1, 1>>(
              p_RETURN_VALUE__568_6_11__f4a_30_15(), p_pt(), p_q__568_6_11__f4a_31_13(), p_delp()),
          gt::make_stage_with_extent<stage__494_func, gt::extent<0, 0, -1, 0>>(
              p_se_mult__568_6_11__f4a_31_13(), p_sw_mult__568_6_11__f4a_31_13()),
          gt::make_stage_with_extent<stage__500_func, gt::extent<0, 0, 0, 1>>(
              p_nw_mult__568_6_11__f4a_31_13(), p_ne_mult__568_6_11__f4a_31_13()),
          gt::make_stage_with_extent<stage__506_func, gt::extent<0, 0, -1, 1>>(
              p_sw_mult__568_6_11__f4a_31_13(), p_pt(), p_q__568_6_11__f4a_31_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__509_func, gt::extent<0, 0, -1, 1>>(
              p_sw_mult__568_6_11__f4a_31_13(), p_pt(), p_q__568_6_11__f4a_31_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__512_func, gt::extent<0, 0, -1, 1>>(
              p_se_mult__568_6_11__f4a_31_13(), p_pt(), p_q__568_6_11__f4a_31_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__515_func, gt::extent<0, 0, -1, 1>>(
              p_se_mult__568_6_11__f4a_31_13(), p_pt(), p_q__568_6_11__f4a_31_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__518_func, gt::extent<0, 0, -1, 1>>(
              p_nw_mult__568_6_11__f4a_31_13(), p_pt(), p_q__568_6_11__f4a_31_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__521_func, gt::extent<0, 0, -1, 1>>(
              p_nw_mult__568_6_11__f4a_31_13(), p_pt(), p_q__568_6_11__f4a_31_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__524_func, gt::extent<0, 0, -1, 1>>(
              p_ne_mult__568_6_11__f4a_31_13(), p_pt(), p_q__568_6_11__f4a_31_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__527_func, gt::extent<0, 0, -1, 1>>(
              p_ne_mult__568_6_11__f4a_31_13(), p_pt(), p_q__568_6_11__f4a_31_13(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__530_func, gt::extent<0, 0, -1, 1>>(
              p_q__568_6_11__f4a_31_13(), p_RETURN_VALUE__568_6_11__f4a_31_13())),
      gt::make_multistage(
          gt::execute::parallel(),
          gt::make_stage_with_extent<stage__533_func, gt::extent<0, 0, -1, 1>>(
              p_RETURN_VALUE__568_6_11__f4a_31_13(), p_w(), p_pt(), p_q__568_6_11__f4a_32_12()),
          gt::make_stage_with_extent<stage__539_func, gt::extent<0, 0, -1, 0>>(
              p_sw_mult__568_6_11__f4a_32_12(), p_se_mult__568_6_11__f4a_32_12()),
          gt::make_stage_with_extent<stage__545_func, gt::extent<0, 0, 0, 1>>(
              p_nw_mult__568_6_11__f4a_32_12(), p_ne_mult__568_6_11__f4a_32_12()),
          gt::make_stage_with_extent<stage__551_func, gt::extent<0, 0, -1, 1>>(
              p_sw_mult__568_6_11__f4a_32_12(), p_w(), p_q__568_6_11__f4a_32_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__554_func, gt::extent<0, 0, -1, 1>>(
              p_sw_mult__568_6_11__f4a_32_12(), p_w(), p_q__568_6_11__f4a_32_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__557_func, gt::extent<0, 0, -1, 1>>(
              p_se_mult__568_6_11__f4a_32_12(), p_w(), p_q__568_6_11__f4a_32_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__560_func, gt::extent<0, 0, -1, 1>>(
              p_se_mult__568_6_11__f4a_32_12(), p_w(), p_q__568_6_11__f4a_32_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__563_func, gt::extent<0, 0, -1, 1>>(
              p_nw_mult__568_6_11__f4a_32_12(), p_w(), p_q__568_6_11__f4a_32_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__566_func, gt::extent<0, 0, -1, 1>>(
              p_nw_mult__568_6_11__f4a_32_12(), p_w(), p_q__568_6_11__f4a_32_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__569_func, gt::extent<0, 0, -1, 1>>(
              p_ne_mult__568_6_11__f4a_32_12(), p_w(), p_q__568_6_11__f4a_32_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__572_func, gt::extent<0, 0, -1, 1>>(
              p_ne_mult__568_6_11__f4a_32_12(), p_w(), p_q__568_6_11__f4a_32_12(),
              p_domain_size_I(), p_domain_size_J(), p_domain_size_K()),
          gt::make_stage_with_extent<stage__575_func, gt::extent<0, 0, -1, 1>>(
              p_q__568_6_11__f4a_32_12(), p_RETURN_VALUE__568_6_11__f4a_32_12())),
      gt::make_multistage(gt::execute::parallel(),
                          gt::make_stage_with_extent<stage__578_func, gt::extent<0, 0, -1, 1>>(
                              p_RETURN_VALUE__568_6_11__f4a_32_12(), p_w()),
                          gt::make_stage_with_extent<stage__581_func, gt::extent<0, 0, 0, 1>>(
                              p_vtc(), p_delp(), p_pt(), p_w(), p_fy1(), p_fy2(), p_fy()),
                          gt::make_stage_with_extent<stage__608_func, gt::extent<0, 0, 0, 0>>(
                              p_delp(), p_fx1(), p_fy1(), p_rarea(), p_pt(), p_fx(), p_fy(),
                              p_delpc(), p_w(), p_fx2(), p_fy2(), p_wc(), p_ptc())));

  gt_computation.run(p_delp() = ds_delp, p_pt() = ds_pt, p_utc() = ds_utc, p_vtc() = ds_vtc,
                     p_w() = ds_w, p_rarea() = ds_rarea, p_delpc() = ds_delpc, p_ptc() = ds_ptc,
                     p_wc() = ds_wc);
  // computation_.sync_bound_data_stores();
}

} // namespace transportdelp____gtx86_d7fde2f45f_pyext