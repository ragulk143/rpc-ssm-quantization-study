        h_uniform = torch.zeros(D_INNER, D_STATE, device=device)
        h_asymm = torch.zeros(D_INNER, D_STATE, device=device)

        print(f" Running profile over {seq_len} tokens on [{device}] (RTX 3050 6GB)...")

        with torch.inference_mode():
            for t in range(seq_len):
                u_t = u_stream[t]

                x_proj_out = u_t @ self.x_proj_weight.t()
                dt_raw, B_t, C_t = torch.split(x_proj_out, [D_MODEL, D_STATE, D_STATE], dim=-1)
                dt_step = F.softplus(dt_raw @ self.dt_proj_weight.t() + self.dt_proj_bias)

                dA_fp16 = torch.exp(dt_step.unsqueeze(-1) * A_fp16.unsqueeze(0))
                dB_fp16 = dt_step.unsqueeze(-1) * B_t.unsqueeze(0)

                # Policy 1: FP16 Baseline
                h_fp16 = dA_fp16 * h_fp16 + (u_t.unsqueeze(-1) @ dB_fp16.mean(dim=0, keepdim=True)).squeeze(0)

                # Policy 2: Uniform 4-bit PTQ
                dA_uni = self.simulate_uniform_4bit(dA_fp16)
                dB_uni = self.simulate_uniform_4bit(dB_fp16)
                u_t_uni = self.simulate_uniform_4bit(u_t)
                h_uniform = dA_uni * h_uniform + (u_t_uni.unsqueeze(-1) @ dB_uni.mean(dim=0, keepdim=True)).squeeze(0)

                # Policy 3: Asymmetric Strategy
                dB_asymm = self.simulate_uniform_4bit(dB_fp16)
                u_t_asymm = self.simulate_uniform_4bit(u_t)
                h_asymm = dA_fp16 * h_asymm + (u_t_asymm.unsqueeze(-1) @ dB_asymm.mean(dim=0, keepdim=True)).squeeze(0)

                if (t + 1) % 2500 == 0:
                    uni_dist = torch.norm(h_fp16 - h_uniform, p='fro').item()
                    asymm_dist = torch.norm(h_fp16 - h_asymm, p='fro').item()
                    print(f"Token Snapshot {t+1:5d} | Uniform PTQ Matrix Drift: {uni_dist:8.2f} | Asymmetric Isolated Drift: {asymm_dist:8.2f}")
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CONTEXT_DEPTH = 15000
    simulated_workload = torch.randn(CONTEXT_DEPTH, D_INNER, device=device)

    profiler = MambaRealTopologyProfiler().to(device)
    profiler.execute_profile(simulated_workload)
    print("\nVerification complete! Data streams successfully calculated.")
