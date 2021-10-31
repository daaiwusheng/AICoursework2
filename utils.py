import torch
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import string
from nltk.translate.bleu_score import sentence_bleu
import re
from torchvision import transforms
from vocabulary import Vocabulary
from config import *
from datasets import Flickr8k_Images, Flickr8k_Features ,Flickr8k_Images_bleu
from random import sample
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_lines(filepath):
    """ Open the ground truth captions into memory, line by line. 
    Args:
        filepath (str): the complete path to the tokens txt file
    """
    file = open(filepath, 'r')
    lines = []

    while True: 
        # Get next line from file 
        line = file.readline() 
        if not line: 
            break
        lines.append(line.strip())
    file.close()
    return lines


def parse_lines(lines):
    """
    Parses token file captions into image_ids and captions.
    Args:
        lines (str list): str lines from token file
    Return:
        image_ids (int list): list of image ids, with duplicates
        cleaned_captions (list of lists of str): lists of words
    """
    image_ids = []
    cleaned_captions = []
    # QUESTION 1.1
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    for line in lines:
        str_image_caption = line.split('.jpg#')
        image_ids.append(str_image_caption[0])
        caption = str_image_caption[1]
        caption = caption[1:len(caption)]
        caption_lower = caption.lower()
        cleaned_caption = caption_lower.strip()
        cleaned_caption = re.sub(remove_chars, '',  cleaned_caption)
        cleaned_captions.append(cleaned_caption)
    return image_ids, cleaned_captions



def build_vocab(cleaned_captions):
    """ 
    Parses training set token file captions and builds a Vocabulary object
    Args:
        cleaned_captions (str list): cleaned list of human captions to build vocab with

    Returns:
        vocab (Vocabulary): Vocabulary object
    """

    # QUESTION 1.1
    # TODO collect words


    # create a vocab instance
    vocab = Vocabulary()

    # add the token words
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # TODO add the rest of the words from the cleaned captions here
    # vocab.add_word('word')
    word_dict = {}
    for sentence in cleaned_captions:
        words = sentence.split()
        for word in words:
            if word in word_dict:
                count = word_dict[word]
                count += 1 #calculate the number of each word appearing in the captions
                word_dict[word] = count
            else:
                word_dict[word] = 1
    for word, value in word_dict.items():
        if value > 3:
            vocab.add_word(word)
    return vocab


def decode_caption(sampled_ids, vocab):
    """ 
    Args:
        sampled_ids (int list): list of word IDs from decoder
        vocab (Vocabulary): vocab for conversion
    Return:
        predicted_caption (str): predicted string sentence
    """
    predicted_caption = ""
    # QUESTION 2.1
    if(torch.is_tensor(sampled_ids)):
        sampled_ids = sampled_ids.cpu().numpy().tolist()
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<start>':
            continue
        if word == '<end>':
            break
        predicted_caption = predicted_caption + " " + word
    return predicted_caption


"""
We need to overwrite the default PyTorch collate_fn() because our 
ground truth captions are sequential data of varying lengths. The default
collate_fn() does not support merging the captions with padding.

You can read more about it here:
https://pytorch.org/docs/stable/data.html#dataloader-collate-fn. 
"""
def caption_collate_fn(data):
    """ Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length from longest to shortest.
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # merge images (from tuple of 3D tensor to 4D tensor).
    # if using features, 2D tensor to 3D tensor. (batch_size, 256)
    images = torch.stack(images, 0) 

    # merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths
#########################################################
#**********helper funcions --------below-----------------
data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),   # using ImageNet norms
                             (0.229, 0.224, 0.225))])

def timshow(x):
    image = np.transpose(x.numpy(),(1,2,0))
    # print(image)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    plt.imshow(image)
    plt.show()
    return image

#encode test images
def get_images_byLoader(image_test_ids):
    dataset_test = Flickr8k_Images(
        image_ids=image_test_ids,
        transform=data_transform,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size= len(image_test_ids),  # change as needed
        shuffle=False,
        num_workers=0,  # may need to set to 0
    )
    dataiter = iter(test_loader)
    images = dataiter.next()
    return images

def encode_test_images(image_test_ids,encoder):
    #1.collect images from image_test_ids
    images = get_images_byLoader(image_test_ids)
    #2.encoder images
    features = []
    images = images.to(device, non_blocking=True)
    with torch.no_grad():
        features_in_batch = encoder(images)
        features_in_batch = torch.squeeze(features_in_batch)
        features_in_batch = features_in_batch.cpu()
        features_in_batch = features_in_batch.numpy().tolist()
    for feature in features_in_batch:
        features.append(feature)
    return features

def get_random_image_ids_and_show(num,test_image_ids):
    #num refers the number of images can be got
    image_test_ids = sample(test_image_ids,num)
    #show images scequencely
    images = get_images_byLoader(image_test_ids)
    # images = images.to(device, non_blocking=True)
    for image in images:
        timshow(image)
    return image_test_ids

def get_specific_image_ids_and_show(image_id):
    images = get_images_byLoader([image_id])
    # images = images.to(device, non_blocking=True)
    for image in images:
        timshow(image)
    image_test_ids = [image_id]
    return image_test_ids

def decoder_image_fetures_to_word_ids(image_features,decoder):
    with torch.no_grad():
        image_features = image_features.to(device, non_blocking=True)
        word_ids = decoder.sample(image_features)
        word_ids = word_ids.to(device, non_blocking=True)
    return word_ids
###################################################################
#functions for bleu socre ---------below-------------
def get_images_loader_bleu(test_image_ids_bleu):
    dataset_test = Flickr8k_Images_bleu(
        image_ids=test_image_ids_bleu,
        transform=data_transform,
    )

    test_loader_bleu = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=128,  # change as needed
        shuffle=False,
        num_workers=0,  # may need to set to 0
    )
    return test_loader_bleu

def encode_test_images_bleu(images,encoder):
    #1.encoder images
    features = []
    images = images.to(device, non_blocking=True)
    with torch.no_grad():
        features_in_batch = encoder(images)
        features_in_batch = torch.squeeze(features_in_batch)
        features_in_batch = features_in_batch.cpu()
        features_in_batch = features_in_batch.numpy().tolist()
    for feature in features_in_batch:
        features.append(feature)
    return features

def decode_caption_bleu(sampled_ids, vocab):
    predicted_caption_list = []
    if(torch.is_tensor(sampled_ids)):
        sampled_ids = sampled_ids.cpu().numpy().tolist()
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<start>':
            continue
        if word == '<end>':
            break
        predicted_caption_list.append(word)
    return predicted_caption_list

def get_dict_test_image_reference_captions(test_image_ids,test_cleaned_captions):
    num_test_captions = len(test_image_ids)
    dict_test_image_captions = {}
    for i in range(num_test_captions):
        image_id = test_image_ids[i]
        reference_caption = test_cleaned_captions[i].split()
        if image_id in dict_test_image_captions.keys():
            captions_list_per_id = dict_test_image_captions[image_id]
            captions_list_per_id.append(reference_caption)
            dict_test_image_captions[image_id] = captions_list_per_id
        else:
            captions_list_per_id = [reference_caption]
            dict_test_image_captions[image_id] = captions_list_per_id
    return dict_test_image_captions

def get_dict_predicted_cations(test_loader_bleu,encoder,decoder,vocab):
    dict_predicted_captions = {}
    for n, data in enumerate(test_loader_bleu):
        images, image_ids_bleu = data
        image_ids_bleu = list(image_ids_bleu)  # tolist
        features_bleu = encode_test_images_bleu(images, encoder)
        features_bleu = torch.tensor(features_bleu)
        word_ids_bleu = decoder_image_fetures_to_word_ids(features_bleu, decoder)
        word_ids_bleu = word_ids_bleu.cpu().numpy().tolist()
        i = 0
        for word_ids_one_sentence_bleu in word_ids_bleu:
            predicted_cap_bleu = decode_caption_bleu(word_ids_one_sentence_bleu, vocab)  # in list
            dict_predicted_captions[image_ids_bleu[i]] = predicted_cap_bleu
            i += 1
    return dict_predicted_captions

#definition kes for bleu socre
HIGHEST_SCORE = "highest_score"
HIGHEST_ID = "highest_id"
LOWEST_SCORE = "lowest_score"
LOWEST_ID = "lowest_id"
OVERALL_AVE_SCORE = "overall_ave_score"
#-------------------------------

def get_bleu_result(test_image_ids_bleu, dict_test_image_captions, dict_predicted_captions):
    dict_result_bleu = {}
    dict_image_id_bleu_socre = {}
    total_socre = 0.0
    highest_score = 0.0
    lowest_score = 1.0
    overall_ave_score = 0.0
    highest_id = 0
    lowest_id = 0
    num_images = len(test_image_ids_bleu)
    for i in range(num_images):
        image_id = test_image_ids_bleu[i]
        reference = dict_test_image_captions[image_id]
        capation = dict_predicted_captions[image_id]
        score = sentence_bleu(reference, capation,
                              smoothing_function=SmoothingFunction().method7) # use default weights
        dict_image_id_bleu_socre[image_id] = score
        total_socre += score
        # get highest socre
        if score > highest_score:
            highest_score = score
            highest_id = image_id
        if score < lowest_score:
            lowest_score = score
            lowest_id = image_id
    overall_ave_score = total_socre / num_images
    dict_result_bleu[OVERALL_AVE_SCORE] = overall_ave_score
    dict_result_bleu[HIGHEST_ID] = highest_id
    dict_result_bleu[HIGHEST_SCORE] = highest_score
    dict_result_bleu[LOWEST_ID] = lowest_id
    dict_result_bleu[LOWEST_SCORE] = lowest_score
    return dict_result_bleu,dict_image_id_bleu_socre

#functions for bleu socre ---above------
################################################################
################################################################
#functions for cosine values ------------below-----------

def get_average_vector(captions_list,vocab_embeddings,vocab):
    current_word_id = vocab.word2idx["<unk>"]
    sum_vector = np.zeros([EMBED_SIZE], dtype="float32")
    average_vector = np.zeros([EMBED_SIZE], dtype="float32")
    for word in captions_list:
        if word in vocab.word2idx:
            current_word_id = vocab.word2idx[word]
        else:
            current_word_id = vocab.word2idx["<unk>"]
        word_embedding = vocab_embeddings[current_word_id]
        sum_vector = np.sum([sum_vector,word_embedding],axis=0)
    average_vector = sum_vector/len(captions_list)
    return average_vector

#definition kes for cosine socre
HIGHEST_COS_VALUE = "highest_cos_value"
HIGHEST_COS_ID = "highest_cos_id"
LOWEST_COS_VALUE = "lowest_cos_value"
LOWEST_COS_ID = "lowest_cos_id"
OVERALL_AVE_COS_VALUE = "overall_ave_cos_value"

def solve_question_2_2_2(vocab_embeddings, dict_predicted_captions, dict_test_image_captions,vocab):
    dict_result = {}
    highest_cos_value = -1
    heigest_cos_image_id = 0
    lowest_cos_value = 1
    lowest_cos_image_id = 0
    image_ids_list = []
    average_cos_values_list = []
    for key_image_id, value_predicted_caption in dict_predicted_captions.items():
        average_vec_pre_cap = get_average_vector(value_predicted_caption, vocab_embeddings,vocab)
        ref_caps_list = dict_test_image_captions[key_image_id]
        average_vec_ref_caps_list = []
        for ref_cap in ref_caps_list:
            average_vec_ref_cap = get_average_vector(ref_cap, vocab_embeddings,vocab)
            average_vec_ref_caps_list.append(average_vec_ref_cap)

        average_vec_ref_caps_list = np.reshape(average_vec_ref_caps_list, [5, EMBED_SIZE])

        cos_value = cosine_similarity(np.expand_dims(average_vec_pre_cap, 0), average_vec_ref_caps_list)
        cos_value = np.squeeze(cos_value)

        average_cos_value = cos_value.sum() / len(cos_value)
        # store image id here, then the id order can be corresponding to average_cos_value in average_cos_values_list
        image_ids_list.append(key_image_id)
        # get highest and lowest cos values and image_ids
        if average_cos_value > highest_cos_value:
            highest_cos_value = average_cos_value
            heigest_cos_image_id = key_image_id
        if average_cos_value < lowest_cos_value:
            lowest_cos_value = average_cos_value
            lowest_cos_image_id = key_image_id

        average_cos_values_list.append(average_cos_value)

    dict_result[HIGHEST_COS_VALUE] = highest_cos_value
    dict_result[HIGHEST_COS_ID] = heigest_cos_image_id
    dict_result[LOWEST_COS_VALUE] = lowest_cos_value
    dict_result[LOWEST_COS_ID] = lowest_cos_image_id
    print(len(average_cos_values_list))
    overall_ave_cos_value = sum(average_cos_values_list) / len(average_cos_values_list)
    dict_result[OVERALL_AVE_COS_VALUE] = overall_ave_cos_value

    return dict_result,average_cos_values_list,image_ids_list

def get_dict_image_id_cos_value(average_cos_values_array, image_ids_cos_list):
    dict_image_id_cos_value = {}
    for i in range(len(image_ids_cos_list)):
        image_id = image_ids_cos_list[i]
        cos_value = average_cos_values_array[i]
        dict_image_id_cos_value[image_id] = cos_value
    return dict_image_id_cos_value

#functions for cosine values ------------above-----------
###############################################################################
#functions for question 2.3 ------------below-----------

SIMILAR_COS_SCORE = 'similar_cos_score'
SIMILAR_BLEU_SCORE = 'similar_bleu_score'
SIMILAR_SCORE_IMAGE_ID = 'similar_score_image_id'

DISSIMILAR_COS_SCORE = 'dissimilar_cos_score'
DISSIMILAR_BLEU_SCORE = 'dissimilar_bleu_score'
DISSIMILAR_SCORE_IMAGE_ID = 'dissimilar_score_image_id'

# dict_image_id_bleu_socre
def get_similar_dissimilar_score_image_id(dict_image_id_cos_value, dict_image_id_bleu_socre):
    dict_similar_dis_score_image_id = {}
    for image_id, cos_value in dict_image_id_cos_value.items():
        bleu_score = dict_image_id_bleu_socre[image_id]
        if abs(cos_value - bleu_score) < 0.001 or abs(cos_value - bleu_score) < 0.0005:
            # these scores are similar
            dict_similar_dis_score_image_id[SIMILAR_SCORE_IMAGE_ID] = image_id
            dict_similar_dis_score_image_id[SIMILAR_COS_SCORE] = cos_value
            dict_similar_dis_score_image_id[SIMILAR_BLEU_SCORE] = bleu_score
        if abs(cos_value - bleu_score) > 0.5 or abs(cos_value - bleu_score) > 0.55:
            # these scores are dissimilar
            dict_similar_dis_score_image_id[DISSIMILAR_SCORE_IMAGE_ID] = image_id
            dict_similar_dis_score_image_id[DISSIMILAR_COS_SCORE] = cos_value
            dict_similar_dis_score_image_id[DISSIMILAR_BLEU_SCORE] = bleu_score
    return dict_similar_dis_score_image_id

#functions for question 2.3 ------------above-----------
###############################################################################

#not that improtant functions, some helpers----below-------

def graph_loss(status_train_recordor):
    fig, ax1 = plt.subplots()
    plt.plot(status_train_recordor, 'b', label = 'training loss')
    plt.legend(loc='center')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('graph loss trend')
    plt.legend(loc='upper left')
    plt.figure(figsize=(10,8),dpi=200)
    plt.show()

#get original reference captions corresponding to image_ids
def store_image_id_captions(lines):
    dict_image_id_captions = {}
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    for line in lines:
        str_image_caption = line.split('.jpg#')
        image_id = str_image_caption[0]
        caption = str_image_caption[1]
        caption = caption[1:len(caption)]
        cleaned_caption = caption.strip()
        cleaned_caption = re.sub(remove_chars, '', cleaned_caption)
        if image_id in dict_image_id_captions.keys():
            captions = dict_image_id_captions[image_id]
            captions.append(cleaned_caption)
            dict_image_id_captions[image_id] = captions
        else:
            captions = [cleaned_caption]
            dict_image_id_captions[image_id] = captions

    return dict_image_id_captions

def get_specific_word_embedding(word,vocab,vocab_embeddings):
    current_word_id = vocab.word2idx[word]
    word_embedding = vocab_embeddings[current_word_id]
    return word_embedding
    
def calculate_2_words_cosine(word_1, word_2,vocab,vocab_embeddings):
    word_1_embedding = get_specific_word_embedding(word_1,vocab,vocab_embeddings)
    word_2_embedding = get_specific_word_embedding(word_2,vocab,vocab_embeddings)
    word_1_vec = np.expand_dims(word_1_embedding, 0)
    word_2_vec = np.expand_dims(word_2_embedding, 0)
    cos_value = cosine_similarity(word_1_vec,word_2_vec)
    return cos_value

