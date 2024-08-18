import torch

class GrokAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
                 alpha_init=0.98, lamb=2.0, gamma=0.1, grokking_signal_fns=None,
                 grokking_signal_decay_rate=0.1, gradient_clipping=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        alpha_init=alpha_init, lamb=lamb, gamma=gamma,
                        grokking_signal_fns=grokking_signal_fns,
                        grokking_signal_decay_rate=grokking_signal_decay_rate,
                        gradient_clipping=gradient_clipping)
        super(GrokAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grokking_signal = self._compute_grokking_signal(group)
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad

                if group['gradient_clipping'] > 0:
                    grad = torch.clamp(grad, -group['gradient_clipping'], group['gradient_clipping'])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['grok_ema'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq, grok_ema = state['exp_avg'], state['exp_avg_sq'], state['grok_ema']
                beta1, beta2 = group['betas']

                state['step'] += 1
                
                # Layer-wise momentum decay
                layer_beta1 = beta1 * (1 - group['gamma'])**i

                # Grokfast component
                alpha = group['alpha_init'] * torch.exp(torch.tensor(-group['grokking_signal_decay_rate'] * grokking_signal))
                grok_ema.mul_(alpha).add_(grad, alpha=1 - alpha)
                grok_grad = grad + group['lamb'] * grok_ema

                # AdamW update with Grokfast-amplified gradient
                exp_avg.mul_(layer_beta1).add_(grok_grad, alpha=1 - layer_beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grok_grad, grok_grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr']

                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
    
    
    def _compute_grokking_signal(self, group):
        if group['grokking_signal_fns'] is None:
            return 0.0

        signals = []
        for fn in group['grokking_signal_fns']:
            try:
                signal = fn()
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                print(f"Error in grokking_signal_fn: {e}. Ignoring this function.")

        if not signals:
            return 0.0

        return sum(signals) / len(signals)