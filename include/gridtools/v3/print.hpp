/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "../common/utility.hpp"

template <class C, class T, class F>
GT_CONSTEXPR GT_FUNCTION auto trinary_op(C &&c, T &&t, F &&f)
    -> decltype(gridtools::wstd::forward<C>(c) ? gridtools::wstd::forward<T>(t) : gridtools::wstd::forward<F>(f)) {
    return gridtools::wstd::forward<C>(c) ? gridtools::wstd::forward<T>(t) : gridtools::wstd::forward<F>(f);
};

namespace gridtools {
    namespace v3 {}
} // namespace gridtools
