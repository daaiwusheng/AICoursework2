import torch
from PIL import Image
from torch.utils.data import Dataset

from config import IMAGE_DIR



class Flickr8k_Images(Dataset):
    """ Flickr8k custom dataset to read image data only,
        compatible with torch.utils.data.DataLoader. """
    
    def __init__(self, image_ids, transform=None):
        """ Set the path for images, captions and vocabulary wrapper.
        
        Args:
            image_ids (str list): list of image ids
            transform: image transformer
        """
        self.image_ids = image_ids
        self.transform = transform


    def __getitem__(self, index):
        """ Returns image. """

        image_id = self.image_ids[index]
        path = IMAGE_DIR + str(image_id) + ".jpg"
        image = Image.open(open(path, 'rb'))

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_ids)


class Flickr8k_Features(Dataset):
    """ Flickr8k custom dataset with features and vocab, compatible with torch.utils.data.DataLoader. """
    
    def __init__(self, image_ids, captions, vocab, features):
        """ Set the path for images, captions and vocabulary wrapper.
        
        Args:
            image_ids (str list): list of image ids
            captions (str list): list of str captions
            vocab: vocabulary wrapper
            features: torch Tensor of extracted features
        """
        self.image_ids = image_ids
        self.captions = captions
        self.vocab = vocab
        self.features = features

    def __getitem__(self, index):
        """ Returns one data pair (feature and target caption). """

        path = IMAGE_DIR + str(self.image_ids[index]) + ".jpg"
        image_features = self.features[index]

        # convert caption (string) to word ids.
        tokens = self.captions[index].split()
        caption = []
        # build the Tensor version of the caption, with token words
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)
        
        return image_features, target

    def __len__(self):
        return len(self.image_ids)

#provide images and image_ids for bleu question
class Flickr8k_Images_bleu(Dataset):
    """ Flickr8k custom dataset to read image data only,
        compatible with torch.utils.data.DataLoader. """

    def __init__(self, image_ids, transform=None):
        """ Set the path for images, captions and vocabulary wrapper.

        Args:
            image_ids (str list): list of image ids
            transform: image transformer
        """
        self.image_ids = image_ids
        self.transform = transform

    def __getitem__(self, index):
        """ Returns image. """

        image_id = self.image_ids[index]
        path = IMAGE_DIR + str(image_id) + ".jpg"
        image = Image.open(open(path, 'rb'))

        if self.transform is not None:
            image = self.transform(image)

        return image,image_id

    def __len__(self):
        return len(self.image_ids)