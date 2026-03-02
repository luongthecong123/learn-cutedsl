import cutlass
import cutlass.cute as cute

@cute.jit
def test():
    tv_layout_tiled = cute.make_layout(
        shape=((4,8,4,4),((2,2),(1,2))),
        stride=((128,1,16,512),((64,8),(0,2048)))
    )
    print("Tiled Layout:")
    print(tv_layout_tiled)
    
    tid = 11
    reg_idx = 3
    lane_coord = cute.idx2crd(tid, tv_layout_tiled.shape[0])
    reg_coord = cute.idx2crd(reg_idx, tv_layout_tiled.shape[1])
    coord = (lane_coord, reg_coord)
    print("coord:", coord)
    
    tile_shape_mnk = (64, 64, 64)
    
    mn_flat = cute.crd2idx(coord, tv_layout_tiled.shape)
    print("mn_flat:", mn_flat)
    
    m, n = cute.idx2crd(mn_flat, (tile_shape_mnk[0], tile_shape_mnk[1]))
    print(f"m={m}, n={n}")

test()