import argparse
import os
#functions to generate features in utils.py
from utils import run_alexnet,run_vgg,run_resnet

def generate_features(image_dir,save_dir,net):

    image_dir_name = image_dir.split("/")[-1]
    feat_save_dir = os.path.join(save_dir,image_dir_name+"_feats")
    net_save_dir = os.path.join(feat_save_dir,net)
    if not os.path.exists(net_save_dir):
        os.makedirs(net_save_dir)

    if net == "alexnet":
        run_alexnet(image_dir,net_save_dir)
    elif net == "vgg":
        run_vgg(image_dir,net_save_dir)
    elif net == "resnet":
        run_resnet(image_dir,net_save_dir)
    else:
        print ("DNN not from list")



def main():
    #dnns list
    dnns_list = ['alexnet','vgg','resnet']

    #ArgumentParser
    parser = argparse.ArgumentParser(description='generate DNN activations from a stimuli dir')
    parser.add_argument('-id','--image_dir', help='stimulus directory path', default = "/media/kshitid20/My Passport1/Algonauts/algonautsChallenge2019/118_Image_Set/118images", type=str)
    parser.add_argument('-sd','--save_dir', help='save directory path', default = "./feats", type=str)
    parser.add_argument("--net",help='DNN choice', default = "vgg", choices=dnns_list)
    args = vars(parser.parse_args())

    image_dir = args['image_dir']
    save_dir = args['save_dir']
    net = args['net']

    #generate features/activations
    generate_features(image_dir,save_dir,net)

if __name__ == "__main__":
    main()
