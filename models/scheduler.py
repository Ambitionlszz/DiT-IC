from diffusers import DPMSolverMultistepScheduler


def make_1step_sched(pretrained_path, device="cuda"):
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        pretrained_path,
        subfolder="scheduler",
    )
    scheduler.set_timesteps(1, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.betas = scheduler.betas.to(device)
    return scheduler


class OneStepFlowScheduler:
    """Apply SANA's fixed one-step flow-prediction update."""

    def __init__(self, base_scheduler):
        self.base_scheduler = base_scheduler

    def step(self, model_output, sample):
        sigma_t = self.base_scheduler.sigmas[0]
        return sample - sigma_t * model_output
