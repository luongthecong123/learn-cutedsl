from cutlass.cute import Swizzle
import cutlass
import cutlass.cute as cute

@cute.jit
def test():
    sC_swizzle = cute.make_swizzle(3, 4, 3)
    sA_layout = cute.make_layout(
        shape=(64,64),
        stride=(64,1)
    )
    sC_layout = cute.make_layout(
        shape=(64, 128),
        stride=(128, 1)
    )
    sC_layout_composed = cute.composition(
        lhs=sA_layout,
        rhs=sC_layout
    )

    sC_swizzled = cute.make_composed_layout(
        inner=sC_swizzle,
        offset=0,
        outer=sC_layout
    )
    
    print("swizzle: ", sC_swizzle)
    print("sC bare: ", sC_layout)
    print("sC conposed: ", sC_layout_composed)
    print("sC swizzed: ", sC_swizzled)

test()