import torch
import torchvision

def get2D_tensor(inp):
    inp = torch.Tensor(inp)
    if len(inp.size()) < 2:
        inp = inp.unsqueeze(0)
    return inp

def XYWH_XYminXYmax(frames):
    frames = get2D_tensor(frames)
    frames[:, 2] += frames[:, 0] - 1
    frames[:, 3] += frames[:, 1] - 1
    return frames

def XYminXYmax_XYWH(frames):
    frames = get2D_tensor(frames)
    frames[:, 2] -= frames[:, 0] - 1
    frames[:, 3] -= frames[:, 1] - 1
    return frames

def formRect(frames, im_sizes):
    frames = get2D_tensor(frames)
    im_sizes = get2D_tensor(im_sizes)
    frames = XYWH_XYminXYmax(frames)
    zero = torch.Tensor([0])
    frames[:, 0] = torch.max(torch.min(frames[:, 0], im_sizes[:, 0]), zero)
    frames[:, 1] = torch.max(torch.min(frames[:, 1], im_sizes[:, 1]), zero)
    frames[:, 2] = torch.max(torch.min(frames[:, 2], im_sizes[:, 0]), zero)
    frames[:, 3] = torch.max(torch.min(frames[:, 3], im_sizes[:, 1]), zero)
    frames = XYminXYmax_XYWH(frames)
    return frames

def transRect(frames, im_sizes):
    frames = get2D_tensor(frames)
    im_sizes = get2D_tensor(im_sizes)
    frames[:, 0] = 2 * frames[:, 0] / im_sizes[:, 0] - 1
    frames[:, 1] = 2 * frames[:, 1] / im_sizes[:, 1] - 1
    frames[:, 2] = 2 * frames[:, 2] / im_sizes[:, 0]
    frames[:, 3] = 2 * frames[:, 3] / im_sizes[:, 1]
    return frames

def transRect_inv(frames, im_sizes):
    frames = get2D_tensor(frames)
    im_sizes = get2D_tensor(im_sizes)
    frames[:, 0] = (frames[:, 0] + 1) / 2 * im_sizes[:, 0]
    frames[:, 1] = (frames[:, 1] + 1) / 2 * im_sizes[:, 1]
    frames[:, 2] = frames[:, 2] / 2 * im_sizes[:, 0]
    frames[:, 3] = frames[:, 3] / 2 * im_sizes[:, 1]
    return frames

def caclArea(frames):
    dx = frames[:, 2] - frames[:, 0]
    dx[dx < 0] = 0
    dy = frames[:, 3] - frames[:, 1]
    dy[dy < 0] = 0
    return dx * dy

def compare(frames1, frames2):
    frames1 = get2D_tensor(frames1)
    frames1 = XYWH_XYminXYmax(frames1)
    frames2 = get2D_tensor(frames2)
    frames2 = XYWH_XYminXYmax(frames2)
    
    intersec = frames1.clone()
    intersec[:, 0] = torch.max(frames1[:, 0], frames2[:, 0])
    intersec[:, 1] = torch.max(frames1[:, 1], frames2[:, 1])
    intersec[:, 2] = torch.min(frames1[:, 2], frames2[:, 2])
    intersec[:, 3] = torch.min(frames1[:, 3], frames2[:, 3])
    
    area1 = caclArea(frames1)
    area2 = caclArea(frames2)
    iarea = caclArea(intersec)
    assert((area1 + area2 - iarea <= 0).sum() == 0)
    
    return iarea  / (area1 + area2 - iarea)    

def calcAccuracy(preds, targets, im_sizes, theta=0.75):
    preds = transRect_inv(preds.clone(), im_sizes)
    preds = formRect(preds, im_sizes)
    targets = transRect_inv(targets.clone(), im_sizes)
    IoU = compare(preds, targets)    
    corr = (IoU >= theta).sum()  
    return float(corr.item() / preds.size(0))

class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt