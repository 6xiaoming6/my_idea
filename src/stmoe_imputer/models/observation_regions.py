"""Disjoint, observed-only F/M/C constraint ledger. No dense incidence matrix."""
import torch
from .observed_consistency_coe import _blocks, _expand_cells


class ObservationRegions:
    def __init__(self, observations):
        # The backbone pyramid contains area-normalized observation densities.
        (xf, mf), (xm, mm), (xc, mc) = [(x.float(), m.bool()) for x, m in observations]
        xf = torch.where(mf, xf, 0.)
        xm = torch.where(mm, xm, 0.) * 4
        xc = torch.where(mc, xc, 0.) * 16
        em = _expand_cells(mm, 2)
        self.owners = [mf, ~mf & em, ~mf & ~em & _expand_cells(mc, 4)]
        self.budgets = [xf, xm - _blocks(xf, 2).sum(-1),
                        xc - _blocks(xm, 2).sum(-1)
                        - _blocks(torch.where(em, 0., xf), 4).sum(-1)]
        self.counts = [_blocks(m.float(), f).sum(-1) for m, f in zip(self.owners, (1, 2, 4))]
        self.budgets = [torch.where(n > 0, b, 0.) for b, n in zip(self.budgets, self.counts)]
        numerator = sum(torch.where(m, x.square(), 0.).sum((2,3,4), keepdim=True)
                        for x,m in [(xf,mf), (xm/4,mm), (xc/16,mc)])
        denominator = sum(m.sum((2,3,4), keepdim=True) for m in (mf,mm,mc))
        self.sigma = (numerator / denominator.clamp_min(1)).sqrt().clamp_min(1).detach()
        self.unconstrained = ~(self.owners[0] | self.owners[1] | self.owners[2])

    def residuals(self, p):
        return [torch.where(n > 0, (b - _blocks(p.float()*m, f).sum(-1))/n.clamp_min(1), 0.)
                for m,b,n,f in zip(self.owners,self.budgets,self.counts,(1,2,4))]

    def project(self, d):
        return sum(_expand_cells(_blocks(d.float()*m, f).sum(-1)/n.clamp_min(1), f)*m
                   for m,n,f in zip(self.owners,self.counts,(1,2,4)))

    def detail(self, d):
        return d.float() - self.project(d)

    def statistics(self, p):
        result=[]
        for e,n,f in zip(self.residuals(p),self.counts,(1,2,4)):
            valid=n>0; count=valid.sum((1,2,3,4)).clamp_min(1)
            v=e/self.sigma
            result.append(torch.stack(((v.abs()*valid).sum((1,2,3,4))/count,
                torch.where(valid.flatten(1).any(1),((v.square()*valid).sum((1,2,3,4))/count).clamp_min(1e-20).sqrt(),0.),
                n.sum((1,2,3,4))/(n[0].numel()*f*f),
                (torch.where(valid,(n-1)/n.clamp_min(1),0.)).sum((1,2,3,4))/count),1))
        return torch.stack(result,1)

    def split_delta(self, previous, current):
        d=current.float()-previous.float(); total=self.project(d)
        before,after=self.residuals(previous),self.residuals(current)
        gain=sum(_expand_cells(((a.abs()-b.abs())/a.abs().clamp_min(1e-6)).clamp(0,1).detach(),f)*m
                 for a,b,m,f in zip(before,after,self.owners,(1,2,4)))
        return total*gain, d-total

    def history_innovation(self, read, current, previous):
        message=self.detail(read-current.detach())
        recent=self.detail(current.detach()-previous.detach())
        axes=(2,3,4); norm=recent.square().sum(axes,keepdim=True)
        coefficient=(message*recent).sum(axes,keepdim=True)/norm.clamp_min(1e-20)
        return message-torch.where(norm>0,coefficient,0.)*recent

    def update(self, old, native, sid):
        if sid == 0:
            return native
        f=(1,2,4)[sid]
        # Shared latent coordinates use the intersection of variable ownership.
        # Thus no variable's visible detail can be overwritten if masks differ.
        owner=self.owners[sid].all(1,keepdim=True)
        n=_blocks(owner.float(),f).sum(-1)
        mean=_blocks(old.float()*owner,f).sum(-1)/n.clamp_min(1)
        return (old.float()+_expand_cells(native.float()-mean,f)*owner).to(old.dtype)
