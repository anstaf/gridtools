#include <gridtools.h>
#include <common/halo_descriptor.h>
#ifdef CUDA_EXAMPLE
#include <stencil-composition/backend_cuda.h>
#else
#include <stencil-composition/backend_block.h>
#include <stencil-composition/backend_naive.h>
#endif

#include <stdlib.h>
#include <stdio.h>

#ifdef CUDA_EXAMPLE
#define BACKEND backend_cuda
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend_block
#else
#define BACKEND backend_naive
#endif
#endif

enum sign {any=-2, minus=-1, zero, plus};
template <sign _I, sign _J, sign _K>
struct direction {
    static const sign I = _I;
    static const sign J = _J;
    static const sign K = _K;
};

#define CALL_DIRECTION(x,y,z)                                           \
    std::cout << "<" << #x << "," << #y << "," << #z << ">" << " calling : "; \
    loop<DataField, direction<x,y,z> >(data_field);
 

namespace gridtools {
    template <typename BoundaryFunction, typename HaloDescriptors = array<halo_descriptor, 3> >
    struct direction_boundary_apply {
        HaloDescriptors halo_descriptors;

        direction_boundary_apply(HaloDescriptors const& hd)
            : halo_descriptors(hd)
        {}

        template <typename DataField, typename Direction>
        void loop(DataField & data_field) const {
            for (int i=halo_descriptors[0].loop_low_bound_outside(Direction::I);
                 i<=halo_descriptors[0].loop_high_bound_outside(Direction::I);
                 ++i) {
                for (int j=halo_descriptors[1].loop_low_bound_outside(Direction::J);
                     j<=halo_descriptors[1].loop_high_bound_outside(Direction::J);
                     ++j) {
                    for (int k=halo_descriptors[2].loop_low_bound_outside(Direction::K);
                         k<=halo_descriptors[2].loop_high_bound_outside(Direction::K);
                         ++k) {
                        typename BoundaryFunction:: template apply<Direction>()(data_field,i,j,k);
                    }
                }
            }
        }

        template <typename DataField>
        void apply(DataField & data_field) const {

            CALL_DIRECTION(minus,minus,minus);
            CALL_DIRECTION(minus,minus, zero);
            CALL_DIRECTION(minus,minus, plus);

            CALL_DIRECTION(minus, zero,minus);
            CALL_DIRECTION(minus, zero, zero);
            CALL_DIRECTION(minus, zero, plus);
	  
            CALL_DIRECTION(minus, plus,minus);
            CALL_DIRECTION(minus, plus, zero);
            CALL_DIRECTION(minus, plus, plus);

            CALL_DIRECTION( zero,minus,minus);
            CALL_DIRECTION( zero,minus, zero);
            CALL_DIRECTION( zero,minus, plus);

            CALL_DIRECTION( zero, zero,minus);
            //     CALL_DIRECTION( zero, zero, zero);
            CALL_DIRECTION( zero, zero, plus);
	  
            CALL_DIRECTION( zero, plus,minus);
            CALL_DIRECTION( zero, plus, zero);
            CALL_DIRECTION( zero, plus, plus);

            CALL_DIRECTION( plus,minus,minus);
            CALL_DIRECTION( plus,minus, zero);
            CALL_DIRECTION( plus,minus, plus);

            CALL_DIRECTION( plus, zero,minus);
            CALL_DIRECTION( plus, zero, zero);
            CALL_DIRECTION( plus, zero, plus);
	  
            CALL_DIRECTION( plus, plus,minus);
            CALL_DIRECTION( plus, plus, zero);
            CALL_DIRECTION( plus, plus, plus);
        }
    };
} // namespace gridtools


// struct bc_input {
//     template <int I, int J, int K> // relative coordinates
//     struct apply {
//         template <typename DataField>
//         void operator()(DataField & data_field, int i, int j, int k) const {
//             printf("General implementation\n");
//             data_field(i,j,k) = -1;
//         }
//     };

//     template <int I, int K> // relative coordinates
//     struct apply<I, -1, K> {
//         template <typename DataField>
//             void operator()(DataField & data_field, int i, int j, int k) const {
//             printf("Implementation going on J upward\n");
//             data_field(i,j,k) = 88;
//         }
//     };

//     template <int K> // relative coordinates
//     struct apply<-1, -1, K> {
//         template <typename DataField>
//             void operator()(DataField & data_field, int i, int j, int k) const {
//             printf("Implementation going on J upward\n");
//             data_field(i,j,k) = 77777;
//         }
//     };
// }


template <typename T>
struct direction_bc_input {

    template <typename AnyDirection, typename Dummy = void> // relative coordinates
    struct apply {
        template <typename DataField>
        void operator()(DataField & data_field, int i, int j, int k) const {
            std::cout << "General implementation AAA" << std::endl;
            data_field(i,j,k) = -1;
        }
    };

    template <sign I, sign K, typename Dummy> // relative coordinates
    struct apply<direction<I, minus, K>, Dummy > {
        template <typename DataField>
            void operator()(DataField & data_field, int i, int j, int k) const {
            std::cout << "Implementation going A-A" << std::endl;
            data_field(i,j,k) = 88;
        }
    };

    template <sign K, typename Dummy> // relative coordinates
    struct apply<direction<minus, minus, K>, Dummy > {
        template <typename DataField>
            void operator()(DataField & data_field, int i, int j, int k) const {
            std::cout << "Implementation going --A" << std::endl;
            data_field(i,j,k) = 77777;
        }
    };

    template <typename Dummy>
    struct apply<direction<minus, minus, minus>, Dummy > {
        template <typename DataField>
            void operator()(DataField & data_field, int i, int j, int k) const {
            std::cout << "Implementation going ---" << std::endl;
            data_field(i,j,k) = 55555;
        }
    };

};



int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: interface1_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    int d1 = atoi(argv[1]);
    int d2 = atoi(argv[2]);
    int d3 = atoi(argv[3]);

    typedef gridtools::BACKEND::storage_type<int, gridtools::layout_map<0,1,2> >::type storage_type;

    // Definition of the actual data fields that are used for input/output
    storage_type in(d1,d2,d3,-1, std::string("in"));
    storage_type out(d1,d2,d3,-7.3, std::string("out"));
    storage_type coeff(d1,d2,d3,8, std::string("coeff"));

    for (int i=0; i<d1; ++i)
        for (int j=0; j<d2; ++j)
            for (int k=0; k<d3; ++k)
                in(i,j,k) = 0.0;

    for (int i=0; i<d1; ++i) {
        for (int j=0; j<d2; ++j) {
            for (int k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }

    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1,1,1,d1-2,d1);
    halos[1] = gridtools::halo_descriptor(1,1,1,d2-2,d2);
    halos[2] = gridtools::halo_descriptor(1,1,1,d3-2,d3);

    gridtools::direction_boundary_apply<direction_bc_input<int> >(halos).apply(in);

    for (int i=0; i<d1; ++i) {
        for (int j=0; j<d2; ++j) {
            for (int k=0; k<d3; ++k) {
                printf("%d ", in(i,j,k));
            }
            printf("\n");
        }
        printf("\n");
    }

    return 0;
}