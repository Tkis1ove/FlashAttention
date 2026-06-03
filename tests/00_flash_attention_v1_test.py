import torch
from flash import flash_attn_v1


def test_flash_attn_v1():
    B, N, H, D = 2, 64, 8, 32
    Q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda")

    O = flash_attn_v1(Q, K, V)

    assert O.shape == Q.shape, f"shape mismatch: {O.shape} vs {Q.shape}"
    assert O.dtype == Q.dtype, f"dtype mismatch: {O.dtype} vs {Q.dtype}"
    assert torch.equal(O, Q), (
        f"kernel 占位逻辑是 O=Q，实际不符: max_diff={(O.float() - Q.float()).abs().max().item()}"
    )

    print(f"[PASS] B={B}, N={N}, H={H}, D={D}, dtype={O.dtype}")


def test_various_shapes():
    shapes = [
        (1, 32, 4, 16),
        (2, 128, 8, 64),
        (4, 256, 12, 32),
    ]
    for B, N, H, D in shapes:
        Q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda")
        K = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda")
        V = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda")

        O = flash_attn_v1(Q, K, V)

        assert O.shape == Q.shape
        assert torch.equal(O, Q), f"failed at shape {B, N, H, D}"
        print(f"[PASS] B={B}, N={N}, H={H}, D={D}")


if __name__ == "__main__":
    test_flash_attn_v1()
    test_various_shapes()
