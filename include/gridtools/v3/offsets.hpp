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

#include <algorithm>

#include "../common/host_device.hpp"
#include "../common/integral_constant.hpp"
#include "../common/literal_parser.hpp"
#include "../meta.hpp"
#include "../stencil/common/dim.hpp"

namespace gridtools {
    namespace v3 {
        template <int...>
        struct offsets {};

        namespace offsets_impl_ {
            namespace lazy {
                template <class>
                struct to_list;

                template <int... Is>
                struct to_list<offsets<Is...>> {
                    using type = meta::list<integral_constant<int, Is>...>;
                };

                template <class>
                struct from_list;

                template <template <class...> class L, class... Ts>
                struct from_list<L<Ts...>> {
                    using type = offsets<Ts::value...>;
                };
            } // namespace lazy
            GT_META_DELEGATE_TO_LAZY(to_list, class T, T);
            GT_META_DELEGATE_TO_LAZY(from_list, class T, T);

            template <class T>
            using is_zero = bool_constant<T::value == 0>;

            template <class Offsets>
            using normalize_offsets = from_list<meta::remove_suffix<is_zero, to_list<Offsets>>>;

            template <class Offset, size_t N, class Lst = to_list<Offset>>
            using complement_offsets = from_list<
                meta::concat<Lst, meta::repeat_c<N - meta::length<Lst>::value, meta::list<integral_constant<int, 0>>>>>;
        } // namespace offsets_impl_
        using offsets_impl_::complement_offsets;
        using offsets_impl_::normalize_offsets;

#define GT_OFFSETS_DEFINE_UNARY_OPERATOR(op)                                                            \
    template <int... Is>                                                                                \
    constexpr GT_FUNCTION normalize_offsets<offsets<(op Is)...>> operator op(offsets<Is...>) noexcept { \
        return {};                                                                                      \
    }                                                                                                   \
    static_assert(1, "")

        GT_OFFSETS_DEFINE_UNARY_OPERATOR(+);
        GT_OFFSETS_DEFINE_UNARY_OPERATOR(-);
        GT_OFFSETS_DEFINE_UNARY_OPERATOR(~);
        GT_OFFSETS_DEFINE_UNARY_OPERATOR(!);

#undef GT_OFFSETS_DEFINE_UNARY_OPERATOR

        template <int... Is, int... Js, std::enable_if_t<sizeof...(Is) == sizeof...(Js), int> = 0>
        constexpr GT_FUNCTION normalize_offsets<offsets<(Is + Js)...>> operator+(offsets<Is...>, offsets<Js...>) {
            return {};
        }

        template <int... Is, int... Js, std::enable_if_t<sizeof...(Is) == sizeof...(Js), int> = 0>
        constexpr GT_FUNCTION normalize_offsets<offsets<(Is - Js)...>> operator-(offsets<Is...>, offsets<Js...>) {
            return {};
        }

        template <int... Is, int... Js, std::enable_if_t<sizeof...(Is) != sizeof...(Js), int> = 0>
        constexpr GT_FUNCTION auto operator+(offsets<Is...>, offsets<Js...>) {
            auto constexpr n = std::max(sizeof...(Is), sizeof...(Js));
            return complement_offsets<offsets<Is...>, n>() + complement_offsets<offsets<Js...>, n>();
        }

        template <int... Is, int... Js, std::enable_if_t<sizeof...(Is) != sizeof...(Js), int> = 0>
        constexpr GT_FUNCTION auto operator-(offsets<Is...>, offsets<Js...>) {
            auto constexpr n = std::max(sizeof...(Is), sizeof...(Js));
            return complement_offsets<offsets<Is...>, n>() - complement_offsets<offsets<Js...>, n>();
        }

    } // namespace v3

    namespace literals {
        template <char... Chars>
        constexpr GT_FUNCTION v3::normalize_offsets<v3::offsets<literal_parser<int, Chars...>::value>> operator"" _i() {
            return {};
        }
        template <char... Chars>
        constexpr GT_FUNCTION v3::normalize_offsets<v3::offsets<0, literal_parser<int, Chars...>::value>>
        operator"" _j() {
            return {};
        }
        template <char... Chars>
        constexpr GT_FUNCTION v3::normalize_offsets<v3::offsets<0, 0, literal_parser<int, Chars...>::value>>
        operator"" _k() {
            return {};
        }
    } // namespace literals
} // namespace gridtools
