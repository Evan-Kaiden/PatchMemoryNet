# https://github.com/valeoai/FOUND

import torch

from train import train
from models.matchnet import Matcher
from samplier import extract_patches
from sim_classifier import sim_classify
from dataloader import trainloader, testloader, memloader


def main():

    def extractor(img):
        return extract_patches(img, kernel_size=12, stride=1)

    device = "mps"
    m = Matcher(10, extractor).to(device)
    opt = torch.optim.Adam(m.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    train(epochs=1, model=m, trainloader=trainloader, testloader=testloader, memloader=memloader, optimizer=opt, criterion=criterion, device=device)
    # sim_classify(m.encoder, testloader, memloader, extractor, device=device)

    torch.save(m.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
