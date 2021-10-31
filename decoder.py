"""
COMP5623M Coursework on Image Caption Generation


python decoder.py


"""

import torch

import numpy as np

import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image

import datetime
import matplotlib.pyplot as plt
from datasets import Flickr8k_Images, Flickr8k_Features
from models import DecoderRNN, EncoderCNN
from utils import *
from config import *
from random import sample
from torch.optim import lr_scheduler

# if false, train model; otherwise try loading model from checkpoint and evaluate
EVAL = True # False#

# reconstruct the captions and vocab, just as in extract_features.py
lines = read_lines(TOKEN_FILE_TRAIN)
image_ids, cleaned_captions = parse_lines(lines)
vocab = build_vocab(cleaned_captions)


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ',device)
#set max_length of one predicted caption
Max_Length = 20

# initialize the models and set the learning parameters
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS,max_seq_length=Max_Length).to(device, non_blocking=True)

print(decoder)

if not EVAL:

    # load the features saved from extract_features.py
    print(len(lines))
    features = torch.load('features.pt', map_location=device)
    print("Loaded features", features.shape)

    features = features.repeat_interleave(5, 0)
    print("Duplicated features", features.shape)

    dataset_train = Flickr8k_Features(
        image_ids=image_ids,
        captions=cleaned_captions,
        vocab=vocab,
        features=features,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64, # change as needed
        shuffle=True,
        num_workers=0, # may need to set to 0
        collate_fn=caption_collate_fn, # explicitly overwrite the collate_fn
    )


    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

    print(len(image_ids))
    print(len(cleaned_captions))
    print(features.shape)


#########################################################################
#
#        QUESTION 1.3 Training DecoderRNN
# 
#########################################################################

    # TODO write training loop on decoder here
    print("current number of epochs = ",NUM_EPOCHS)
    status_train_recordor = []

    loss_train_data = 0.0
    
    for epoch in range(NUM_EPOCHS):
        starttime_per_epoch = datetime.datetime.now()
        running_loss = 0.0
        n = 0
        for i, data in enumerate(train_loader):
            image_features,target_captions,lengths_list = data
            lengths_list_tensor = torch.tensor(lengths_list)
            # move to device
            image_features = image_features.to(device, non_blocking=True)
            target_captions = target_captions.to(device, non_blocking=True)
            optimizer.zero_grad()

            outputs = decoder(image_features,target_captions, lengths_list_tensor)
            outputs = outputs.to(device, non_blocking=True)
            targets = pack_padded_sequence(target_captions, lengths_list_tensor, batch_first=True)[0]
            targets = targets.to(device, non_blocking=True)

            loss = criterion(outputs, targets)
            loss = loss.to(device, non_blocking=True)

            loss.backward()
            optimizer.step()
            # accumulate loss
            running_loss += loss.item()
            n += 1

        loss_train_data = running_loss / n
        status_train_recordor.append(loss_train_data) 
        print(f"current epoch: {epoch} " 
              + f"              loss : {loss_train_data : .3f}")
        endtime_per_epoch = datetime.datetime.now()
        print(f"time consuming per epoch：{(endtime_per_epoch - starttime_per_epoch).total_seconds() : .3f} ")
    # for each batch, prepare the targets using this torch.nn.utils.rnn function
    # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

    # save model after training
    graph_loss(status_train_recordor)
    decoder_ckpt = torch.save(decoder, "decoder.ckpt")



# if we already trained, and EVAL == True, reload saved model
else:
    test_lines = read_lines(TOKEN_FILE_TEST)
    test_image_ids, test_cleaned_captions = parse_lines(test_lines)
    # get original ref captions with respective image_id
    dict_image_id_test_captions_original = store_image_id_captions(test_lines)
    # load models
    encoder = EncoderCNN().to(device)
    if device.type == 'cuda':
        decoder = torch.load("decoder.ckpt").to(device) #on colab
    else:
        decoder = torch.load("decoder.ckpt",map_location=torch.device('cpu')).to(device) #on macbook pro
    encoder.eval()
    decoder.eval() # generate caption, eval mode to not influence batchnorm
#########################################################################
#
#        QUESTION 2.1 Generating predictions on test data
# 
#########################################################################
    # TODO define decode_caption() function in utils.py
    numbers_would_like_to_get = 1   # set any number for getting random images or set it as 1 when get specific image by id

    def solve_question_2_1(numbers_would_like_to_get,specific_image_id):
        image_test_ids = []
        if numbers_would_like_to_get == 1:
            # get specific image id
            image_test_ids = get_specific_image_ids_and_show(specific_image_id)
        else:
            # get random image id
            image_test_ids = get_random_image_ids_and_show(numbers_would_like_to_get,test_image_ids)

        features = encode_test_images(image_test_ids,encoder)
        features = torch.tensor(features)
        if len(image_test_ids) == 1:
            features = torch.unsqueeze(features, 0)
        word_ids = decoder_image_fetures_to_word_ids(features,decoder)
        word_ids = word_ids.cpu().numpy().tolist()
        # TODO define decode_caption() function in utils.py
        if numbers_would_like_to_get > 1:
            i = 0
            for word_ids_one_sentence in word_ids:
                predicted_caption = decode_caption(word_ids_one_sentence, vocab)
                image_id = image_test_ids[i]
                print(image_id + "--VV--")
                print("predicted_caption : ",predicted_caption)
                print("reference captions : ",dict_image_id_test_captions_original[image_id])
                i += 1
        else:
            word_ids = word_ids[0]
            predicted_caption = decode_caption(word_ids, vocab)
            image_id = image_test_ids[0]
            print(image_id + " --VV--")
            print("predicted_caption : ", predicted_caption)
            print("reference captions : ", dict_image_id_test_captions_original[image_id])

    solve_question_2_1(numbers_would_like_to_get,"756909515_a416161656")

#########################################################################
#
#        QUESTION 2.2-3 Caption evaluation via text similarity 
# 
#########################################################################
    ####################################
    #--QUESTION 2.2.1--
    ####################################
    # Feel free to add helper functions to utils.py as needed,
    # documenting what they do in the code and in your report

    #1.pair image_id with reference caption
    dict_test_image_captions = get_dict_test_image_reference_captions(test_image_ids,test_cleaned_captions)
    #2.prepare image_ids with every id single in
    test_image_ids_bleu = test_image_ids[::5]
    test_loader_bleu = get_images_loader_bleu(test_image_ids_bleu) #data loader for loading test images

    #3.generate predicted_captions in a dict, which key is image_id and value is one's prediction
    dict_predicted_captions = get_dict_predicted_cations(test_loader_bleu,encoder,decoder,vocab)

    #4.get the overall average socre and the highest one and the lowest one
    #get targeted values by the following type
    #dict[OVERALL_AVE_SCORE]
    #dict[HIGHEST_ID] dict[LOWEST_ID]
    #dict[HIGHEST_SCORE] dict[LOWEST_SCORE]

    dict_result_bleu,dict_image_id_bleu_socre = get_bleu_result(test_image_ids_bleu,dict_test_image_captions,dict_predicted_captions)#open when submit

    # #Save
    np.save('dict_result_bleu.npy', dict_result_bleu)
    # #Load
    # dict_result_bleu = np.load('dict_result_bleu.npy',allow_pickle=True).item()

    print("BLEU result :")
    for k,v in dict_result_bleu.items():
        print(k,v)

    print("hightest id and prediction and references : ",
          dict_result_bleu[HIGHEST_ID],"\n",
          ' '.join(dict_predicted_captions[dict_result_bleu[HIGHEST_ID]]),"\n",
          dict_test_image_captions[dict_result_bleu[HIGHEST_ID]]
          )

    print("lowest id and prediction and references : ",
          dict_result_bleu[LOWEST_ID],"\n",
          ' '.join(dict_predicted_captions[dict_result_bleu[LOWEST_ID]]),"\n",
          dict_test_image_captions[dict_result_bleu[LOWEST_ID]]
          )

    ####################################
    # --QUESTION 2.2.2--
    ####################################
    #1.get the vocab word embeddings
    vocab_embeddings = decoder.embed.weight
    print(vocab_embeddings.shape)
    #2.calculate every cosine value of every image's predicted caption and its reference captions.
    #Then store cosin values which are needed for question 2.2.2
    #Then store them in a dict
    #the keys are these
    #HIGHEST_COS_VALUE HIGHEST_COS_ID LOWEST_COS_VALUE LOWEST_COS_ID OVERALL_AVE_COS_VALUE
    print(device,253, type(device))
    print(device.type=='cuda')
    if device.type == 'cuda':
        vocab_embeddings = vocab_embeddings.detach().cpu().numpy().tolist() #run on cloab
    else:
        vocab_embeddings = vocab_embeddings.cpu().numpy().tolist() #run on macbookpro

    dict_cos_similarity,average_cos_values_list,image_ids_cos_list  = solve_question_2_2_2(vocab_embeddings,dict_predicted_captions,dict_test_image_captions,vocab)
    #Save
    np.save('dict_cos_similarity.npy', dict_cos_similarity)
    #Load
    # dict_cos_similarity = np.load('dict_cos_similarity.npy',allow_pickle=True).item()

    print("cosine similarity result : ")
    for k,v in dict_cos_similarity.items():
        print(k,v)

    print("overall average cosine value : ", dict_cos_similarity[OVERALL_AVE_COS_VALUE])
    highest_image_id = dict_cos_similarity[HIGHEST_COS_ID]
    print("highest cosine-value predicted caption : ", ' '.join(dict_predicted_captions[highest_image_id]))
    print("highest cosine-value reference captions : ", dict_image_id_test_captions_original[highest_image_id])
    lowest_image_id = dict_cos_similarity[LOWEST_COS_ID]
    print("lowest cosine-value predicted caption : ", ' '.join(dict_predicted_captions[lowest_image_id]))
    print("lowest cosine-value reference captions : ", dict_image_id_test_captions_original[lowest_image_id])

    ####################################
    # --QUESTION 2.3.1--
    ####################################
    #use all cosine values in average_cos_values_list to normalize, then project these values to 0-1
    #and then calculate the new overall cosine value
    average_cos_values_array = np.array(average_cos_values_list)
    average_cos_values_array = (average_cos_values_array - np.min(average_cos_values_array))/(np.max(average_cos_values_array) - np.min(average_cos_values_array))
    # average_cos_values_normlized = np.sum(average_cos_values_array)/len(average_cos_values_array)
    average_cos_values_normlized = (average_cos_values_array.sum())/len(average_cos_values_array)
    print('average_cos_values_normlized : ',average_cos_values_normlized)

    ####################################
    # --QUESTION 2.3.2--
    ####################################
    dict_image_id_cos_value = get_dict_image_id_cos_value(average_cos_values_array,image_ids_cos_list)
    dict_similar_dis_score_image_id = get_similar_dissimilar_score_image_id(dict_image_id_cos_value,dict_image_id_bleu_socre)
    for k,v in dict_similar_dis_score_image_id.items():
        print(k,v)

    print("similar socre captions:")
    print('similar predicted caption : ',dict_predicted_captions[dict_similar_dis_score_image_id[SIMILAR_SCORE_IMAGE_ID]])
    print('similar reference caption : ',dict_test_image_captions[dict_similar_dis_score_image_id[SIMILAR_SCORE_IMAGE_ID]])
    print("dissimilar socre captions:")
    print('dissimilar predicted caption : ',dict_predicted_captions[dict_similar_dis_score_image_id[DISSIMILAR_SCORE_IMAGE_ID]])
    print('dissimilar reference caption : ',dict_test_image_captions[dict_similar_dis_score_image_id[DISSIMILAR_SCORE_IMAGE_ID]])



