import torchvision.datasets as datasets

voc2007 = datasets.VOCDetection(root="./data", year="2007", image_set="train", download=True)
voc2012 = datasets.VOCDetection(root="./data", year="2012", image_set="train", download=True)
