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

#include <type_traits>

namespace gridtools {
    namespace literal_parser_impl_ {
        template <int Base>
        constexpr int to_int(char c);

        template <>
        constexpr int to_int<2>(char c) {
            return c == '0' ? 0 : c == '1' ? 1 : throw "invalid binary literal";
        }
        template <>
        constexpr int to_int<8>(char c) {
            return c >= '0' && c <= '7' ? c - '0' : throw "invalid octal _c literal";
        }
        template <>
        constexpr int to_int<10>(char c) {
            return c >= '0' && c <= '9' ? c - '0' : throw "invalid decimal _c literal";
        }
        template <>
        constexpr int to_int<16>(char c) {
            return c >= 'A' && c <= 'F'
                       ? c - 'A' + 10
                       : c >= 'a' && c <= 'f' ? c - 'a' + 10
                                              : c >= '0' && c <= '9' ? c - '0' : throw "invalid hex _c literal";
        }

        template <class T, int Base>
        constexpr T parse(char const *first, char const *last) {
            return *last == '\''
                       ? parse<T, Base>(first, last - 1)
                       : T{to_int<Base>(*last)} + (first == last ? 0 : parse<T, Base>(first, last - 1) * Base);
        }

        template <class T, int Base, char... Chars>
        struct digits_parser {
            constexpr static char digits[sizeof...(Chars)] = {Chars...};
            constexpr static T value = parse<T, Base>(digits, digits + sizeof...(Chars) - 1);
        };

        template <class T, char... Chars>
        struct literal_parser : digits_parser<T, 10, Chars...> {};

        template <class T>
        struct literal_parser<T, '0'> : std::integral_constant<T, 0> {};

        template <class T, char... Chars>
        struct literal_parser<T, '0', Chars...> : digits_parser<T, 8, Chars...> {};

        template <class T, char... Chars>
        struct literal_parser<T, '0', 'x', Chars...> : digits_parser<T, 16, Chars...> {};

        template <class T, char... Chars>
        struct literal_parser<T, '0', 'b', Chars...> : digits_parser<T, 2, Chars...> {};
    } // namespace literal_parser_impl_

    using literal_parser_impl_::literal_parser;
} // namespace gridtools
