function generate_feats_resnet(image_dir,save_dir)
% generates resnet activations of the selected layers
% Inputs:
% image_dir : directory containing images in .jpg format
% save_dir :  directory to save the activations
% The activations are saved in the format image_name.mat and contains
% activation of the selected layers of DNN when the image_name.jpg was
% given as input

% change the path to your own matconvnet path
run /home/kshitid20/Documents/Work_with_Radek/matconvnet-1.0-beta25/matlab/vl_setupnn

% path to save the downloaded resnet weights
modelPath = 'data/models/imagenet-resnet-50-dag.mat' ;

% path to save the downloaded resnet weights
if ~exist(modelPath)
  mkdir(fileparts(modelPath)) ;
  urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat', ...
    modelPath) ;
end

% loading network
net = dagnn.DagNN.loadobj(load(modelPath)) ;

% displays and creates directory to save activations
disp(save_dir)
mkdir(save_dir);

% Getting image file lists in the image directory
imagefiles = dir(strcat(image_dir,'/*.jpg'));      
nfiles = length(imagefiles);  

% selecting layers for saving activations
% to get list of all possible layers add a breakpoint here and 
% check net.layers in matlab variables 
select_layers_list = ["res2c","res3d","res4f","res5c"];

%for loop to iterate over all image files and generate DNN activations
for ii=1:nfiles
    % reading image file
    currentfilename = imagefiles(ii).name;
    file_path = strcat(imagefiles(ii).folder,'/',imagefiles(ii).name);
    currentimage = imread(file_path);
    im_ = single(currentimage) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - net.meta.normalization.averageImage ;

    % parameter to keep all the activations in memory
    net.conserveMemory = 0;
   
    % getting layer_index for the selected layers
    if ii==1
        num_layers = size(net.layers);
        num_selected_layers = size(select_layers_list);
        for layer=1:num_selected_layers(2)
            for jj=1:num_layers(2)
                if select_layers_list(layer)==net.layers(jj).name
                    layer_index(layer) = jj+1;
                end
            end
        end
    end

    % forward pass
    net.eval({'data', im_});
    
    % save paths
    save_file_name = strsplit(currentfilename,".");
    save_file_name = strcat(save_file_name(1),".mat");
    save_file_path = strcat(save_dir,'/',save_file_name);

    % assigning layer activations to matlab variables    
    res2c = net.vars(layer_index(1)).value;
    res3d = net.vars(layer_index(2)).value;
    res4f = net.vars(layer_index(3)).value;
    res5c = net.vars(layer_index(4)).value;

    save(save_file_path,'res2c','res3d','res4f','res5c')
end


