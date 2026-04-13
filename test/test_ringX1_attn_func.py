import os
import torch
import torch.distributed as dist

try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False
    from ringX_attn.fused_attention import attention as fused_attention_func

from ringX_attn.ringX1_attn import ringX1_attn_func as ringX_attn_func
from utils import log, set_seed


if __name__ == "__main__":
    device = torch.device(f"xpu:{int(os.environ['PALS_RANKID']) % int(os.environ['PALS_LOCAL_SIZE'])}")
    torch.xpu.set_device(device.index or 0)

    dist.init_process_group(
        backend="xccl",
        init_method="env://",
        world_size=int(os.environ["PALS_WORLD_SIZE"]),
        rank=int(os.environ["PALS_RANKID"]),
    )
    rank = int(os.environ["PALS_RANKID"])
    world_size = int(os.environ["PALS_WORLD_SIZE"])

    set_seed(rank)
    dtype = torch.float16

    batch_size = 1
    seqlen = 6144 ## divisible by 12 and 128 and 384
    num_heads = 5
    head_dim = 128
    dropout_p = 0
    causal = False
    deterministic = False
    softmax_scale = head_dim ** -0.5

    assert not causal
    assert seqlen % world_size == 0
    assert head_dim % 8 == 0

    q = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    with torch.no_grad():
        dist.broadcast(q, src=0)
        dist.broadcast(k, src=0)
        dist.broadcast(v, src=0)

    dout = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    with torch.no_grad():
        dist.broadcast(dout, src=0)

    local_q = q.chunk(world_size, dim=1)[rank].detach().clone()
    local_k = k.chunk(world_size, dim=1)[rank].detach().clone()
    local_v = v.chunk(world_size, dim=1)[rank].detach().clone()
    local_q.requires_grad = True
    local_k.requires_grad = True
    local_v.requires_grad = True

    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    if HAS_FLASH:
        out, lse, _ = flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    else:
        # fused_attention_func expects (batch, heads, seqlen, head_dim)
        # test uses (batch, seqlen, heads, head_dim) — transpose in and out
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        out_t, lse = fused_attention_func(q_t, k_t, v_t, causal, softmax_scale)
        out = out_t.transpose(1, 2)

    local_out = out.chunk(world_size, dim=1)[rank]
    local_lse = lse.chunk(world_size, dim=-1)[rank]

    ring_out, ring_lse, _ = ringX_attn_func(
        local_q, local_k, local_v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
        group=dist.group.WORLD,
    )

    log("out", out, rank0_only=True)
    log("lse", lse, rank0_only=True)
    log("out diff", local_out - ring_out)
    log("lse diff", local_lse - ring_lse)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    out.backward(dout)
    dq = q.grad
    dk = k.grad
    dv = v.grad
    local_dq = dq.chunk(world_size, dim=1)[rank]
    local_dk = dk.chunk(world_size, dim=1)[rank]
    local_dv = dv.chunk(world_size, dim=1)[rank]

    ring_out.backward(local_dout)
    ring_dq = local_q.grad
    ring_dk = local_k.grad
    ring_dv = local_v.grad

    log("dq diff", local_dq[:] - ring_dq[:])
    log("dk diff", local_dk[:] - ring_dk[:])
    log("dv diff", local_dv[:] - ring_dv[:])
