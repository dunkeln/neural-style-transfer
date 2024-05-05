from functools import reduce
import torch


def compose(*fns):
    def __inner__(*args, **kwargs):
        return reduce(lambda x, f: f(x), fns, *args, **kwargs)
    return __inner__



def whiten_and_color(cF, sF):
    cFSize = cF.size()
    # Compute mean and subtract
    c_mean = torch.mean(cF.view(cFSize[0], -1), dim=1, keepdim=True)
    cF = cF - c_mean.view(cFSize[0], 1, 1)
    # Compute content covariance
    cF_flat = cF.view(cFSize[0], -1)
    contentConv = torch.mm(cF_flat, cF_flat.t()) / (cF_flat.size(1) - 1) + torch.eye(cFSize[0], dtype=cF.dtype, device=cF.device)
    # Singular value decomposition
    c_u, c_e, c_v = torch.svd(contentConv, some=False)
    # Determine truncation point
    k_c = torch.sum(c_e > 0.00001).item()

    # Whiten content feature
    c_d = torch.rsqrt(c_e[:k_c])
    step1 = torch.mm(c_v[:, :k_c], torch.diag(c_d))
    step2 = torch.mm(step1, c_v[:, :k_c].t())
    whiten_cF = torch.mm(step2, cF_flat)
    # Compute mean and subtract for style feature
    sFSize = sF.size()
    s_mean = torch.mean(sF.view(sFSize[0], -1), dim=1, keepdim=True)
    sF = sF - s_mean.view(sFSize[0], 1, 1)
    # Compute style covariance
    sF_flat = sF.view(sFSize[0], -1)
    styleConv = torch.mm(sF_flat, sF_flat.t()) / (sF_flat.size(1) - 1)
    # Singular value decomposition
    s_u, s_e, s_v = torch.svd(styleConv, some=False)
    # Determine truncation point
    k_s = torch.sum(s_e > 0.00001).item()
    # Recolor content feature
    s_d = torch.sqrt(s_e[:k_s])
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, :k_s], torch.diag(s_d)), s_v[:, :k_s].t()), whiten_cF)
    targetFeature = targetFeature + s_mean.view(sFSize[0], 1, 1).expand_as(targetFeature)
    return targetFeature

def transform(self,cF,sF,csF,alpha):
    cF = cF.double()
    sF = sF.double()
    C,W,H = cF.size(0),cF.size(1),cF.size(2)
    _,W1,H1 = sF.size(0),sF.size(1),sF.size(2)
    cFView = cF.view(C,-1)
    sFView = sF.view(C,-1)

    targetFeature = self.whiten_and_color(cFView,sFView)
    targetFeature = targetFeature.view_as(cF)
    ccsF = alpha * targetFeature + (1.0 - alpha) * cF
    ccsF = ccsF.float().unsqueeze(0)
    csF.data.resize_(ccsF.size()).copy_(ccsF)
    return csF