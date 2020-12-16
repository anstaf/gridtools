/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/v3/print.hpp>

#include <string>

#include <gtest/gtest.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/v3/offsets.hpp>

using namespace gridtools;
using namespace v3;
using namespace literals;

namespace grammar {
    template <class Field, class Offset>
    struct access {};

    template <class, class>
    struct plus {};
    template <class, class>
    struct minus {};
    template <class, class>
    struct mul {};
    template <class, class>
    struct div {};
    template <class, class>
    struct less {};
    template <class, class>
    struct le {};
    template <class, class>
    struct eq {};
    template <class, class>
    struct or_ {};
    template <class, class>
    struct and_ {};
    template <class>
    struct not_ {};
    template <class, class, class>
    struct if_ {};

    template <class Lhs, class Rhs>
    constexpr plus<Lhs, Rhs> operator+(Lhs, Rhs) {
        return {};
    }
    template <class Lhs, class Rhs>
    constexpr minus<Lhs, Rhs> operator-(Lhs, Rhs) {
        return {};
    }
    template <class Lhs, class Rhs>
    constexpr mul<Lhs, Rhs> operator*(Lhs, Rhs) {
        return {};
    }
    template <class Lhs, class Rhs>
    constexpr div<Lhs, Rhs> operator/(Lhs, Rhs) {
        return {};
    }
    template <class Lhs, class Rhs>
    constexpr less<Lhs, Rhs> operator<(Lhs, Rhs) {
        return {};
    }
    template <class Lhs, class Rhs>
    constexpr le<Lhs, Rhs> operator<=(Lhs, Rhs) {
        return {};
    }
    template <class Lhs, class Rhs>
    constexpr eq<Lhs, Rhs> operator==(Lhs, Rhs) {
        return {};
    }
    template <class Lhs, class Rhs>
    constexpr not_<le<Rhs, Lhs>> operator>(Lhs, Rhs) {
        return {};
    }
    template <class Lhs, class Rhs>
    constexpr not_<less<Rhs, Lhs>> operator>=(Lhs, Rhs) {
        return {};
    }
    template <class Lhs, class Rhs>
    constexpr not_<eq<Lhs, Rhs>> operator!=(Lhs, Rhs) {
        return {};
    }
    template <class Lhs, class Rhs>
    constexpr and_<Lhs, Rhs> operator&&(Lhs, Rhs) {
        return {};
    }
    template <class Lhs, class Rhs>
    constexpr or_<Lhs, Rhs> operator||(Lhs, Rhs) {
        return {};
    }
    template <class T>
    constexpr not_<T> operator!(T) {
        return {};
    }
    template <class C, class T, class F>
    constexpr if_<C, T, F> trinary_op(C const &, T const &, F const &) {
        return {};
    }

    template <size_t N>
    struct arg {
        constexpr access<arg, offsets<>> operator()() const { return {}; }
        template <int... Is>
        constexpr access<arg, offsets<Is...>> operator()(offsets<Is...>) const {
            return {};
        }
    };

    template <class>
    struct tmp {
        constexpr access<tmp, offsets<>> operator()() const { return {}; }
        template <int... Is>
        constexpr access<tmp, offsets<Is...>> operator()(offsets<Is...>) const {
            return {};
        }
    };

} // namespace grammar

inline constexpr auto d = [](auto offset) { return [=](auto f) { return f(offset) - f(); }; };

inline constexpr auto cap = [](auto g, auto f) { return trinary_op(f * g > 0_c, 0_c, f); };

inline constexpr auto flax = [](auto offset) {
    return [=](auto in, auto lap) { return cap(d(offset)(in), d(offset)(lap)); };
};

inline constexpr auto hd = [](auto lift) {
    return [lap = lift([](auto in) { return 4_c * in() - in(1_i) - in(-1_i) - in(1_j) - in(-1_j); }),
               hd = [flx = lift(flax(1_i)), fly = lift(flax(1_j)), out = [](auto in, auto coeff, auto flx, auto fly) {
                   return in() - coeff() * (flx() - flx(-1_i) + fly() - fly(-1_j));
               }](auto in, auto coeff, auto lap) {
                   return out(in, coeff, flx(in, lap), fly(in, lap));
               }](auto in, auto coeff) { return hd(in, coeff, lap(in)); };
};

inline constexpr auto hd2 = [](auto lift) {
    auto tmp0 = lift([](auto a0) { return 4_c * a0() - a0(1_i) - a0(-1_i) - a0(1_j) - a0(-1_j); });
    auto tmp1 = lift(flax(1_i));
    auto tmp2 = lift(flax(1_j));
    return [=](auto a0, auto a1) {
        auto t0 = tmp0(a0);
        auto t1 = tmp1(a0, t0);
        auto t2 = tmp2(a0, t0);
        return a0() - a1() * (t1() - t1(-1_i) + t2() - t2(-1_j));
    };
};

constexpr auto x = hd([](auto f) { return [=](auto... args) { return grammar::tmp<decltype(f(args...))>{}; }; })(
    grammar::arg<0>(), grammar::arg<1>());

GT_META_PRINT_TYPE(decltype(x));

TEST(print, hori_diff) {}
